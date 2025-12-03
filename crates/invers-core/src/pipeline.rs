//! Image processing pipeline
//!
//! Core pipeline for negative-to-positive conversion.

use crate::config;
use crate::decoders::DecodedImage;
use crate::models::{BaseEstimation, BaseEstimationMethod, ConvertOptions};
use crate::verbose_println;
use rayon::prelude::*;

#[cfg(feature = "gpu")]
use crate::gpu;

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

    /// Whether to export as grayscale (single channel)
    /// Set true for B&W images to save space (1 channel instead of 3)
    pub export_as_grayscale: bool,
}

/// Execute the full processing pipeline
pub fn process_image(
    image: DecodedImage,
    options: &ConvertOptions,
) -> Result<ProcessedImage, String> {
    // Try GPU path if enabled and available
    #[cfg(feature = "gpu")]
    if options.use_gpu && gpu::is_gpu_available() {
        if options.debug {
            if let Some(info) = gpu::gpu_info() {
                eprintln!("[DEBUG] Using GPU acceleration: {}", info);
            }
        }

        match gpu::process_image_gpu(&image, options) {
            Ok(result) => return Ok(result),
            Err(e) => {
                // Fall back to CPU on GPU error
                eprintln!("[WARN] GPU processing failed, falling back to CPU: {}", e);
            }
        }
    }

    #[cfg(feature = "gpu")]
    if options.use_gpu && !gpu::is_gpu_available() && options.debug {
        eprintln!("[DEBUG] GPU requested but not available, using CPU");
    }

    // CPU path
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
        source_is_grayscale,
        is_monochrome,
        ..
    } = image;

    // Track if we should export as grayscale (source was grayscale or detected as monochrome)
    let export_as_grayscale = source_is_grayscale || is_monochrome;

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
        let params = if options.no_clip {
            // No-clip mode: normalize without stretching highlights beyond original max
            crate::auto_adjust::auto_levels_no_clip(&mut data, channels, options.auto_levels_clip_percent)
        } else {
            crate::auto_adjust::auto_levels(&mut data, channels, options.auto_levels_clip_percent)
        };

        if options.debug {
            eprintln!(
                "[DEBUG] After auto-levels{} (clip={:.1}%) - R:[{:.4}-{:.4}], G:[{:.4}-{:.4}], B:[{:.4}-{:.4}]",
                if options.no_clip { " (no-clip)" } else { "" },
                options.auto_levels_clip_percent,
                params[0], params[1], params[2], params[3], params[4], params[5]
            );
            let stats = compute_stats(&data);
            eprintln!(
                "[DEBUG]   min: {:.6}, max: {:.6}, mean: {:.6}",
                stats.0, stats.1, stats.2
            );
        }

        // Apply headroom preservation if requested (skip in no-clip mode as it's redundant)
        if options.preserve_headroom && !options.no_clip {
            apply_headroom(&mut data);
            if options.debug {
                let stats = compute_stats(&data);
                eprintln!(
                    "[DEBUG] After headroom preservation - min: {:.6}, max: {:.6}, mean: {:.6}",
                    stats.0, stats.1, stats.2
                );
            }
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

    // Step 3.15: Apply auto white balance (if enabled)
    // Note: Auto-WB always applies full correction - it needs to boost channels for proper
    // color balance. The no-clip mode only affects other operations.
    if options.enable_auto_wb {
        let multipliers = crate::auto_adjust::auto_white_balance(&mut data, channels, options.auto_wb_strength);

        if options.debug {
            eprintln!(
                "[DEBUG] After auto-wb - multipliers: R={:.4}, G={:.4}, B={:.4}",
                multipliers[0], multipliers[1], multipliers[2]
            );
            let stats = compute_stats(&data);
            eprintln!(
                "[DEBUG]   min: {:.6}, max: {:.6}, mean: {:.6}",
                stats.0, stats.1, stats.2
            );
        }
    }

    // Step 3.2: Apply auto-color (neutralize color casts)
    // Skip if auto-wb was applied - they do similar things and shouldn't be combined
    if options.enable_auto_color && !options.enable_auto_wb {
        let adjustments = if options.no_clip {
            // No-clip mode: limit gains to prevent exceeding current max
            crate::auto_adjust::auto_color_no_clip(
                &mut data,
                channels,
                options.auto_color_strength,
                options.auto_color_min_gain,
                options.auto_color_max_gain,
            )
        } else {
            crate::auto_adjust::auto_color(
                &mut data,
                channels,
                options.auto_color_strength,
                options.auto_color_min_gain,
                options.auto_color_max_gain,
            )
        };

        if options.debug {
            eprintln!(
                "[DEBUG] After auto-color{} (strength={:.2}) - adjustments: R={:.4}, G={:.4}, B={:.4}",
                if options.no_clip { " (no-clip)" } else { "" },
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
        let gain = if options.no_clip {
            crate::auto_adjust::auto_exposure_no_clip(
                &mut data,
                options.auto_exposure_target_median,
                options.auto_exposure_strength,
                options.auto_exposure_min_gain,
                options.auto_exposure_max_gain,
            )
        } else {
            crate::auto_adjust::auto_exposure(
                &mut data,
                options.auto_exposure_target_median,
                options.auto_exposure_strength,
                options.auto_exposure_min_gain,
                options.auto_exposure_max_gain,
            )
        };

        if options.debug {
            let stats = compute_stats(&data);
            eprintln!(
                "[DEBUG] Auto exposure{} (target={:.3}) gain={:.4} - min: {:.6}, max: {:.6}, mean: {:.6}",
                if options.no_clip { " (no-clip)" } else { "" },
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
        if options.no_clip {
            // In no-clip mode, only allow exposure reduction (gain <= 1.0)
            let safe_exposure = options.exposure_compensation.min(1.0);
            if safe_exposure < 1.0 {
                for value in data.iter_mut() {
                    *value = *value * safe_exposure;
                }
            }
            if options.debug && options.exposure_compensation > 1.0 {
                eprintln!(
                    "[DEBUG] Exposure compensation {:.2}x skipped in no-clip mode (would clip)",
                    options.exposure_compensation
                );
            }
        } else {
            for value in data.iter_mut() {
                let scaled = *value * options.exposure_compensation;
                *value = clamp_to_working_range(scaled);
            }
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
    // In no-clip mode, skip color matrix as it can push values outside range
    // In MaskAware mode, skip color matrix as the mask correction handles color balance
    let skip_matrix_for_mask_aware =
        options.inversion_mode == crate::models::InversionMode::MaskAware;

    if !options.skip_color_matrix && !options.no_clip && !skip_matrix_for_mask_aware {
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
        if options.no_clip {
            eprintln!("[DEBUG] Color matrix skipped (no-clip mode)");
        } else if skip_matrix_for_mask_aware {
            eprintln!("[DEBUG] Color matrix skipped (MaskAware mode handles color balance)");
        } else {
            eprintln!("[DEBUG] Color matrix skipped");
        }
    }

    // Step 5: Apply tone curve (unless skipped)
    // In no-clip mode, skip tone curve as it can clip values
    if !options.skip_tone_curve && !options.no_clip {
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
        if options.no_clip {
            eprintln!("[DEBUG] Tone curve skipped (no-clip mode)");
        } else {
            eprintln!("[DEBUG] Tone curve skipped");
        }
    }

    // Step 6: Apply scan profile adjustments (only if a scan profile is specified)
    if let Some(ref scan_profile) = options.scan_profile {
        // Apply per-channel gamma if specified
        if let Some(gamma) = scan_profile.default_gamma {
            if gamma != [1.0, 1.0, 1.0] {
                for pixel in data.chunks_exact_mut(3) {
                    pixel[0] = pixel[0].powf(1.0 / gamma[0]);
                    pixel[1] = pixel[1].powf(1.0 / gamma[1]);
                    pixel[2] = pixel[2].powf(1.0 / gamma[2]);
                }

                if options.debug {
                    let stats = compute_stats(&data);
                    eprintln!(
                        "[DEBUG] After scan profile gamma [{:.2}, {:.2}, {:.2}] - min: {:.6}, max: {:.6}, mean: {:.6}",
                        gamma[0], gamma[1], gamma[2], stats.0, stats.1, stats.2
                    );
                }
            }
        }

        // Apply HSL adjustments if specified
        if let Some(ref hsl_adj) = scan_profile.hsl_adjustments {
            if hsl_adj.has_adjustments() {
                crate::color::apply_hsl_adjustments(&mut data, hsl_adj);

                if options.debug {
                    let stats = compute_stats(&data);
                    eprintln!(
                        "[DEBUG] After scan profile HSL adjustments - min: {:.6}, max: {:.6}, mean: {:.6}",
                        stats.0, stats.1, stats.2
                    );
                }
            }
        }
    }

    // Step 7: Colorspace transform (for now, keep in same space)
    // TODO: Implement colorspace transforms in M3

    // Final guard: keep values within photographic working range
    // Skip in no-clip mode to preserve full dynamic range
    if !options.no_clip {
        enforce_working_range(&mut data);
    } else if options.debug {
        let stats = compute_stats(&data);
        eprintln!(
            "[DEBUG] Final (no-clip mode) - min: {:.6}, max: {:.6}, mean: {:.6}",
            stats.0, stats.1, stats.2
        );
    }

    // Return processed image
    Ok(ProcessedImage {
        width,
        height,
        data,
        channels,
        export_as_grayscale,
    })
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
    }
}

/// Estimate film base from a manually specified ROI
fn estimate_base_from_manual_roi(
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
        num_brightest, percentage
    );
    verbose_println!(
        "[BASE]   medians=[{:.6}, {:.6}, {:.6}] noise=[{:.5}, {:.5}, {:.5}]",
        medians[0], medians[1], medians[2], noise_stats[0], noise_stats[1], noise_stats[2]
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

/// Filter out clipped/extreme pixels that are not valid film base
///
/// Film base should be bright but NOT clipped white. This filters:
/// - Near-white clipped pixels (all channels > 0.95)
/// - Very dark pixels (all channels < 0.05)
/// - Bright grayscale pixels without color variation (not orange mask)
/// Filter results from base pixel filtering
struct FilteredBasePixels {
    /// Valid pixels that passed filtering
    pixels: Vec<[f32; 3]>,
    /// Ratio of pixels that were clipped (0.0-1.0)
    clipped_ratio: f32,
    /// Ratio of pixels that were too dark (0.0-1.0)
    dark_ratio: f32,
}

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

/// Estimate film base using discrete border regions (top, bottom, left, right)
fn estimate_base_from_regions(image: &DecodedImage) -> Result<BaseEstimation, String> {
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
                candidate.label, width, height, x, y
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
            candidate.label, candidate.brightness, num_brightest, percentage
        );
        verbose_println!(
            "[BASE]   medians=[{:.6}, {:.6}, {:.6}] noise=[{:.5}, {:.5}, {:.5}]",
            medians[0], medians[1], medians[2], noise_stats[0], noise_stats[1], noise_stats[2]
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
                    candidate.label, candidate.brightness
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

#[derive(Clone, Copy, Debug)]
struct BaseRoiCandidate {
    rect: (u32, u32, u32, u32),
    brightness: f32,
    label: &'static str,
}

impl BaseRoiCandidate {
    fn new(rect: (u32, u32, u32, u32), brightness: f32, label: &'static str) -> Self {
        Self {
            rect,
            brightness,
            label,
        }
    }

    fn from_manual_roi(image: &DecodedImage, rect: (u32, u32, u32, u32)) -> Self {
        let brightness = sample_region_brightness(image, rect.0, rect.1, rect.2, rect.3);
        Self {
            rect,
            brightness,
            label: "manual",
        }
    }
}

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
        verbose_println!("[BASE]   detected B&W film (low chroma), skipping orange mask validation");
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
                    pixels.push([
                        image.data[idx],
                        image.data[idx + 1],
                        image.data[idx + 2],
                    ]);
                }
            }
        }
    }

    pixels
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
fn estimate_base_from_border(
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
        return Err("No valid film base pixels found in border (all clipped or extreme)".to_string());
    }

    let (num_brightest, percentage, medians, noise_stats) =
        compute_base_stats(&filtered.pixels, sample_fraction);

    verbose_println!(
        "[BASE] Border | using {} px ({:.1}%) brightest",
        num_brightest, percentage
    );
    verbose_println!(
        "[BASE]   medians=[{:.6}, {:.6}, {:.6}] noise=[{:.5}, {:.5}, {:.5}]",
        medians[0], medians[1], medians[2], noise_stats[0], noise_stats[1], noise_stats[2]
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
fn estimate_base_from_histogram(image: &DecodedImage) -> Result<BaseEstimation, String> {
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
        let min_ch = r.min(g).min(b);

        if max_ch >= MIN_BRIGHT && min_ch <= MAX_BRIGHT {
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

        for bin in min_bin..=max_bin {
            if hist[bin] > peak_count {
                peak_count = hist[bin];
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
            // Tighter range (1.35-2.2) to exclude pixels with elevated blue
            // from scanner artifacts or edge effects
            let gb_ratio = g / b;
            (1.35..=2.2).contains(&gb_ratio)
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
        candidates[0].0, candidates[1].0, candidates[2].0, candidates[3].0
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
        crate::models::InversionMode::DivideBlend => {
            // Divide blend inversion:
            // 1. Divide: pixel / base (per channel)
            // 2. Apply gamma 2.2 (like Photoshop's Exposure layer)
            // 3. Invert: 1.0 - result
            //
            // This mimics the Photoshop workflow:
            // - Create layer filled with base color
            // - Set blend mode to "Divide"
            // - Add Exposure adjustment layer at 2.2 gamma
            // - Add inversion via Linear Light
            const GAMMA: f32 = 1.0 / 2.2;

            for pixel in data.chunks_exact_mut(3) {
                // Step 1: Divide
                let divided_r = (pixel[0] / base_r).clamp(0.0, 10.0);
                let divided_g = (pixel[1] / base_g).clamp(0.0, 10.0);
                let divided_b = (pixel[2] / base_b).clamp(0.0, 10.0);

                // Step 2: Apply gamma 2.2 (linear to gamma-encoded)
                let gamma_r = divided_r.powf(GAMMA);
                let gamma_g = divided_g.powf(GAMMA);
                let gamma_b = divided_b.powf(GAMMA);

                // Step 3: Invert
                pixel[0] = 1.0 - gamma_r;
                pixel[1] = 1.0 - gamma_g;
                pixel[2] = 1.0 - gamma_b;
            }
        }
        crate::models::InversionMode::MaskAware => {
            // Orange mask-aware inversion for color negative film.
            //
            // This mode properly accounts for the orange mask that exists in color
            // negative film due to dye impurities. The mask adds constant dye to
            // shadows (clear areas of the negative). When inverted naively, these
            // orange shadows become light blue instead of true black.
            //
            // Algorithm from Observable notebook "Why is Color Negative Film Orange?"
            // by Evan Dorsky (https://observablehq.com/@dorskyee/understanding-color-film)
            //
            // Steps:
            // 1. Perform standard inversion: 1.0 - (pixel / base)
            // 2. Calculate per-channel shadow floor from mask characteristics
            // 3. Apply shadow correction: corrected = (value - floor) / (1 - floor)

            // Get mask profile (auto-detected or use default)
            let mask_profile = base
                .mask_profile
                .clone()
                .unwrap_or_else(crate::models::MaskProfile::default);

            // Calculate shadow floor values
            let (_red_floor, green_floor, blue_floor) = mask_profile.calculate_shadow_floors();

            if options.debug {
                eprintln!(
                    "[DEBUG] MaskAware inversion: magenta_impurity={:.3}, cyan_impurity={:.3}",
                    mask_profile.magenta_impurity, mask_profile.cyan_impurity
                );
                eprintln!(
                    "[DEBUG] Shadow floors: green={:.4}, blue={:.4}",
                    green_floor, blue_floor
                );
            }

            // Step 1: Standard inversion with division (similar to DivideBlend but without gamma)
            for pixel in data.chunks_exact_mut(3) {
                pixel[0] = 1.0 - (pixel[0] / base_r);
                pixel[1] = 1.0 - (pixel[1] / base_g);
                pixel[2] = 1.0 - (pixel[2] / base_b);
            }

            // Step 2: Apply per-channel shadow correction
            // This removes the blue cast by shifting the shadow floor to black
            // Red channel: no correction needed (yellow coupler doesn't absorb red)
            // Green channel: compensate for cyan's green absorption
            // Blue channel: compensate for magenta's blue absorption
            for pixel in data.chunks_exact_mut(3) {
                // Green correction
                if green_floor > 0.0 {
                    pixel[1] = (pixel[1] - green_floor) / (1.0 - green_floor);
                }
                // Blue correction
                if blue_floor > 0.0 {
                    pixel[2] = (pixel[2] - blue_floor) / (1.0 - blue_floor);
                }
            }
        }
        crate::models::InversionMode::BlackAndWhite => {
            // Simple B&W inversion with headroom
            //
            // For grayscale/monochrome images, we do a straightforward inversion:
            // 1. Simple inversion: base - pixel
            // 2. Scale so that (base - headroom) maps to white (1.0)
            //
            // This sets the black point slightly below the film base to preserve
            // shadow detail near the base density.

            // Default 5% headroom
            const BW_HEADROOM: f32 = 0.05;

            // Use average base for B&W (all channels should be similar)
            let base = (base_r + base_g + base_b) / 3.0;

            // Effective black point: base minus headroom fraction
            let black_point = base * (1.0 - BW_HEADROOM);

            // Scale factor: 1.0 / black_point to map black_point -> 1.0
            let scale = 1.0 / black_point.max(0.0001);

            if options.debug {
                eprintln!(
                    "[DEBUG] B&W inversion: base={:.4}, black_point={:.4}, scale={:.4}",
                    base, black_point, scale
                );
            }

            for pixel in data.chunks_exact_mut(3) {
                // Simple inversion and scale
                pixel[0] = ((base - pixel[0]) * scale).clamp(0.0, 1.0);
                pixel[1] = ((base - pixel[1]) * scale).clamp(0.0, 1.0);
                pixel[2] = ((base - pixel[2]) * scale).clamp(0.0, 1.0);
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
        "asymmetric" => {
            // Apply asymmetric film-like curve with separate toe/shoulder controls
            apply_asymmetric_curve(data, curve_params);
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
///
/// Uses parallel processing for large images (>100k values)
fn apply_s_curve(data: &mut [f32], strength: f32) {
    // Clamp strength to valid range
    let strength = strength.clamp(0.0, 1.0);

    if strength < 0.01 {
        // Effectively linear, no transformation needed
        return;
    }

    const PARALLEL_THRESHOLD: usize = 300_000; // 100k pixels * 3 channels

    if data.len() >= PARALLEL_THRESHOLD {
        // Parallel processing for large images
        const CHUNK_SIZE: usize = 256 * 3;
        data.par_chunks_mut(CHUNK_SIZE).for_each(|chunk| {
            for value in chunk.iter_mut() {
                *value = apply_s_curve_point(*value, strength);
            }
        });
    } else {
        // Sequential for small images
        for value in data.iter_mut() {
            *value = apply_s_curve_point(*value, strength);
        }
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

/// Apply asymmetric film-like tone curve
///
/// This curve has three distinct regions:
/// - Toe (shadows): Lifts shadows using a gamma-like curve
/// - Mid (linear): Passes through unchanged for natural midtones
/// - Shoulder (highlights): Compresses highlights using soft-clip
///
/// The result is more film-like than symmetric S-curves because real film
/// has different response characteristics in shadows vs highlights.
///
/// Uses parallel processing for large images (>100k values)
fn apply_asymmetric_curve(data: &mut [f32], params: &crate::models::ToneCurveParams) {
    let strength = params.strength.clamp(0.0, 1.0);

    if strength < 0.01 {
        return; // Effectively linear
    }

    // Clamp parameters to valid ranges
    let toe_strength = params.toe_strength.clamp(0.0, 1.0);
    let shoulder_strength = params.shoulder_strength.clamp(0.0, 1.0);
    let toe_length = params.toe_length.clamp(0.05, 0.45);
    let shoulder_start = params.shoulder_start.clamp(0.55, 0.95);

    const PARALLEL_THRESHOLD: usize = 300_000; // 100k pixels * 3 channels

    if data.len() >= PARALLEL_THRESHOLD {
        // Parallel processing for large images
        const CHUNK_SIZE: usize = 256 * 3;
        data.par_chunks_mut(CHUNK_SIZE).for_each(|chunk| {
            for value in chunk.iter_mut() {
                let curved = apply_asymmetric_curve_point(
                    *value,
                    toe_strength,
                    shoulder_strength,
                    toe_length,
                    shoulder_start,
                );
                *value = clamp_to_working_range(*value * (1.0 - strength) + curved * strength);
            }
        });
    } else {
        // Sequential for small images
        for value in data.iter_mut() {
            let curved = apply_asymmetric_curve_point(
                *value,
                toe_strength,
                shoulder_strength,
                toe_length,
                shoulder_start,
            );
            *value = clamp_to_working_range(*value * (1.0 - strength) + curved * strength);
        }
    }
}

/// Apply asymmetric curve transformation to a single value
///
/// Implements a piecewise curve:
/// - x < toe_length: Toe region with shadow lift (gamma < 1)
/// - toe_length <= x <= shoulder_start: Linear passthrough
/// - x > shoulder_start: Shoulder region with highlight compression
fn apply_asymmetric_curve_point(
    x: f32,
    toe_strength: f32,
    shoulder_strength: f32,
    toe_length: f32,
    shoulder_start: f32,
) -> f32 {
    let x = x.clamp(0.0, 1.0);

    if x < toe_length {
        // Toe region: lift shadows
        // Use power function: output = toe_length * (x / toe_length)^(1/gamma)
        // where gamma > 1 for shadow lift (we use 1/(1 + toe_strength))
        let gamma = 1.0 / (1.0 + toe_strength * 1.5);
        let normalized = x / toe_length;
        let lifted = normalized.powf(gamma);

        // Scale back to toe_length and apply smooth transition
        // The output at toe_length should equal toe_length for continuity
        toe_length * lifted
    } else if x > shoulder_start {
        // Shoulder region: compress highlights
        // Use soft-clip: output = shoulder_start + (1 - shoulder_start) * (1 - (1 - t)^gamma)
        // where t = (x - shoulder_start) / (1 - shoulder_start)
        let gamma = 1.0 + shoulder_strength * 2.0;
        let range = 1.0 - shoulder_start;
        let normalized = (x - shoulder_start) / range;
        let compressed = 1.0 - (1.0 - normalized).powf(gamma);

        // Scale back to remaining range
        shoulder_start + range * compressed
    } else {
        // Linear mid region: pass through unchanged
        x
    }
}

/// Apply color correction matrix
/// Performs 3x3 matrix multiplication on RGB pixels
///
/// Uses parallel processing for large images (>100k pixels)
pub fn apply_color_matrix(data: &mut [f32], matrix: &[[f32; 3]; 3], channels: u8) {
    if channels != 3 {
        return; // Only works for RGB
    }

    let num_pixels = data.len() / 3;
    const PARALLEL_THRESHOLD: usize = 100_000; // Use parallelism for >100k pixels

    if num_pixels >= PARALLEL_THRESHOLD {
        // Parallel processing for large images
        // Process in chunks of 256 pixels (768 f32s) for good cache locality
        const CHUNK_SIZE: usize = 256 * 3;
        data.par_chunks_mut(CHUNK_SIZE).for_each(|chunk| {
            for pixel in chunk.chunks_exact_mut(3) {
                apply_color_matrix_to_pixel(pixel, matrix);
            }
        });
    } else {
        // Sequential processing for small images
        for pixel in data.chunks_exact_mut(3) {
            apply_color_matrix_to_pixel(pixel, matrix);
        }
    }
}

/// Apply color matrix to a single pixel
#[inline(always)]
fn apply_color_matrix_to_pixel(pixel: &mut [f32], matrix: &[[f32; 3]; 3]) {
    let r = pixel[0];
    let g = pixel[1];
    let b = pixel[2];

    // Matrix multiplication: output = matrix * input
    pixel[0] = clamp_to_working_range(matrix[0][0] * r + matrix[0][1] * g + matrix[0][2] * b);
    pixel[1] = clamp_to_working_range(matrix[1][0] * r + matrix[1][1] * g + matrix[1][2] * b);
    pixel[2] = clamp_to_working_range(matrix[2][0] * r + matrix[2][1] * g + matrix[2][2] * b);
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

/// Apply headroom preservation
///
/// Remaps output from 0-1 range to approximately 0.005-0.98 range
/// to preserve shadow and highlight detail.
///
/// Characteristics:
/// - Minimum luminance: ~0.005 (lifted shadows)
/// - Maximum luminance: ~0.98 (preserved highlights)
fn apply_headroom(data: &mut [f32]) {
    const OUTPUT_BLACK: f32 = 0.005;
    const OUTPUT_WHITE: f32 = 0.98;
    const OUTPUT_RANGE: f32 = OUTPUT_WHITE - OUTPUT_BLACK;

    for value in data.iter_mut() {
        // Remap from 0-1 to OUTPUT_BLACK-OUTPUT_WHITE
        *value = OUTPUT_BLACK + (*value * OUTPUT_RANGE);
    }
}

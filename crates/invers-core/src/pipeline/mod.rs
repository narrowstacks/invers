//! Image processing pipeline
//!
//! Core pipeline for negative-to-positive conversion.
//!
//! This module is organized into submodules:
//! - `base_estimation`: Film base detection and measurement
//! - `inversion`: Negative-to-positive conversion algorithms
//! - `tone_mapping`: Tone curves and contrast adjustments

mod base_estimation;
mod inversion;
mod tone_mapping;

// Re-export public items from submodules
pub use base_estimation::{
    estimate_base, estimate_base_from_border, estimate_base_from_histogram,
    estimate_base_from_manual_roi, estimate_base_from_regions, BaseRoiCandidate,
};
pub use inversion::{apply_reciprocal_inversion, invert_negative};
pub use tone_mapping::{apply_asymmetric_curve, apply_log_curve, apply_s_curve, apply_tone_curve};

use crate::decoders::DecodedImage;
use crate::models::{
    BaseEstimation, ConvertOptions, DensityBalance, NeutralPointSample, PipelineMode,
};
use rayon::prelude::*;

#[cfg(feature = "gpu")]
use crate::gpu;

/// Prevent values from ever hitting absolute black/white while retaining full range.
const WORKING_RANGE_FLOOR: f32 = 1.0 / 65535.0;
const WORKING_RANGE_CEILING: f32 = 1.0 - WORKING_RANGE_FLOOR;

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
///
/// Routes to the appropriate pipeline based on `options.pipeline_mode`:
/// - Legacy: Original Invers pipeline with multiple inversion modes
/// - Research: Density-balance-first pipeline for eliminating color crossover
/// - CbStyle: Curve-based pipeline inspired by Negative Lab Pro
pub fn process_image(
    image: DecodedImage,
    options: &ConvertOptions,
) -> Result<ProcessedImage, String> {
    match options.pipeline_mode {
        PipelineMode::Legacy => process_image_legacy(image, options),
        PipelineMode::Research => process_image_research(image, options),
        PipelineMode::CbStyle => crate::cb_pipeline::process_image_cb(image, options),
    }
}

/// Legacy processing pipeline (original Invers implementation)
///
/// Pipeline stages:
/// 1. Base subtraction and inversion (Linear, Logarithmic, DivideBlend, MaskAware, or BlackAndWhite)
/// 2. Shadow lift and highlight compression
/// 3. Auto-levels (histogram stretching)
/// 4. Film base offsets
/// 5. Auto white balance
/// 6. Auto-color
/// 7. Auto-exposure
/// 8. Exposure compensation
/// 9. Color matrix
/// 10. Tone curve
/// 11. Scan profile gamma and HSL adjustments
fn process_image_legacy(
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
            crate::auto_adjust::auto_levels_no_clip(
                &mut data,
                channels,
                options.auto_levels_clip_percent,
            )
        } else {
            // Use auto_levels_with_mode to support different stretching modes
            // Unified mode uses same stretch for all channels to preserve color relationships
            crate::auto_adjust::auto_levels_with_mode(
                &mut data,
                channels,
                options.auto_levels_clip_percent,
                options.auto_levels_mode,
            )
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
        let multipliers =
            crate::auto_adjust::auto_white_balance(&mut data, channels, options.auto_wb_strength);

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

    // Step 3.2: Apply auto-color (scene-adaptive color correction)
    // Can run after auto-WB - they complement each other:
    // - Auto-WB corrects based on highlights (what should be white)
    // - Auto-color refines midtones (scene-adaptive correction)
    if options.enable_auto_color {
        let adjustments = if options.no_clip {
            // No-clip mode: limit gains to prevent exceeding current max
            crate::auto_adjust::auto_color_no_clip(
                &mut data,
                channels,
                options.auto_color_strength,
                options.auto_color_min_gain,
                options.auto_color_max_gain,
                options.auto_color_max_divergence,
            )
        } else {
            crate::auto_adjust::auto_color(
                &mut data,
                channels,
                options.auto_color_strength,
                options.auto_color_min_gain,
                options.auto_color_max_gain,
                options.auto_color_max_divergence,
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
                    *value *= safe_exposure;
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

// ============================================================================
// Research Pipeline Implementation
// ============================================================================

/// Research-based processing pipeline implementing densitometry principles.
///
/// Key innovation: **Density balance BEFORE inversion** using per-channel power
/// functions to align characteristic curves. This eliminates color crossover
/// between shadows and highlights.
///
/// Pipeline stages:
/// 1. Film base white balance (divide by base to normalize orange mask)
/// 2. Density balance (per-channel power: R^db_r, G^1.0, B^db_b)
/// 3. Reciprocal inversion (positive = k / negative)
/// 4. Auto-levels (histogram normalization)
/// 5. Tone curve
/// 6. Export
///
/// Reference: research.md - "The secret to accurate color negative conversion
/// lies not in simple inversion, but in per-channel density balance"
fn process_image_research(
    image: DecodedImage,
    options: &ConvertOptions,
) -> Result<ProcessedImage, String> {
    // TODO: GPU path for research pipeline
    #[cfg(feature = "gpu")]
    if options.use_gpu && gpu::is_gpu_available() && options.debug {
        eprintln!("[DEBUG] Research pipeline GPU not yet implemented, using CPU");
        // Future: match gpu::process_image_research_gpu(&image, options) { ... }
    }

    process_image_research_cpu(image, options)
}

/// CPU implementation of the research pipeline
fn process_image_research_cpu(
    image: DecodedImage,
    options: &ConvertOptions,
) -> Result<ProcessedImage, String> {
    // Step 1: Get base estimation
    let base_estimation = match &options.base_estimation {
        Some(base) => base.clone(),
        None => {
            return Err(
                "Base estimation required. Use analyze-base command first or provide --roi"
                    .to_string(),
            );
        }
    };

    // Step 2: Extract image data
    let DecodedImage {
        width,
        height,
        channels,
        mut data,
        source_is_grayscale,
        is_monochrome,
        ..
    } = image;

    let export_as_grayscale = source_is_grayscale || is_monochrome;

    if channels != 3 {
        return Err(format!("Pipeline requires 3-channel RGB, got {}", channels));
    }

    if options.debug {
        eprintln!("[RESEARCH] Starting research pipeline");
        let stats = compute_stats(&data);
        eprintln!(
            "[RESEARCH] Input - min: {:.6}, max: {:.6}, mean: {:.6}",
            stats.0, stats.1, stats.2
        );
    }

    // Stage 1: Film base white balance (divide by base to normalize orange mask)
    apply_film_base_white_balance(&mut data, &base_estimation, options)?;

    // Stage 2: Density balance (THE CRITICAL STEP - per-channel power BEFORE inversion)
    let density_balance = get_or_compute_density_balance(&data, width, height, options)?;
    apply_density_balance(&mut data, &density_balance, options)?;

    // Stage 3: Reciprocal inversion (positive = k / negative)
    apply_reciprocal_inversion(&mut data, options)?;

    // Stage 4: Auto-levels (histogram normalization)
    if options.enable_auto_levels {
        let params = if options.no_clip {
            crate::auto_adjust::auto_levels_no_clip(
                &mut data,
                channels,
                options.auto_levels_clip_percent,
            )
        } else {
            crate::auto_adjust::auto_levels_with_mode(
                &mut data,
                channels,
                options.auto_levels_clip_percent,
                options.auto_levels_mode,
            )
        };

        if options.debug {
            eprintln!(
                "[RESEARCH] After auto-levels - R:[{:.4}-{:.4}], G:[{:.4}-{:.4}], B:[{:.4}-{:.4}]",
                params[0], params[1], params[2], params[3], params[4], params[5]
            );
        }
    }

    // Stage 4.5: Auto white balance (optional post-processing)
    if options.enable_auto_wb {
        use crate::models::AutoWbMode;

        let multipliers = match options.auto_wb_mode {
            AutoWbMode::GrayPixel => crate::auto_adjust::auto_white_balance(
                &mut data,
                channels,
                options.auto_wb_strength,
            ),
            AutoWbMode::Average => crate::auto_adjust::auto_white_balance_avg(
                &mut data,
                channels,
                options.auto_wb_strength,
            ),
            AutoWbMode::Percentile => {
                // Use 98th percentile for robust white patch
                crate::auto_adjust::auto_white_balance_percentile(
                    &mut data,
                    channels,
                    options.auto_wb_strength,
                    98.0,
                )
            }
        };

        if options.debug {
            eprintln!(
                "[RESEARCH] After auto-WB ({:?}) - multipliers: R={:.4}, G={:.4}, B={:.4}",
                options.auto_wb_mode, multipliers[0], multipliers[1], multipliers[2]
            );
            let stats = compute_stats(&data);
            eprintln!(
                "[RESEARCH]   min: {:.6}, max: {:.6}, mean: {:.6}",
                stats.0, stats.1, stats.2
            );
        }
    }

    // Stage 4.6: Auto-color (scene-adaptive color correction)
    if options.enable_auto_color {
        let adjustments = if options.no_clip {
            crate::auto_adjust::auto_color_no_clip(
                &mut data,
                channels,
                options.auto_color_strength,
                options.auto_color_min_gain,
                options.auto_color_max_gain,
                options.auto_color_max_divergence,
            )
        } else {
            crate::auto_adjust::auto_color(
                &mut data,
                channels,
                options.auto_color_strength,
                options.auto_color_min_gain,
                options.auto_color_max_gain,
                options.auto_color_max_divergence,
            )
        };

        if options.debug {
            eprintln!(
                "[RESEARCH] After auto-color (strength {:.2}) - gains: R={:.4}, G={:.4}, B={:.4}",
                options.auto_color_strength, adjustments[0], adjustments[1], adjustments[2]
            );
            let stats = compute_stats(&data);
            eprintln!(
                "[RESEARCH]   min: {:.6}, max: {:.6}, mean: {:.6}",
                stats.0, stats.1, stats.2
            );
        }
    }

    // Stage 5: Exposure compensation
    if (options.exposure_compensation - 1.0).abs() > 0.001 {
        for value in data.iter_mut() {
            *value *= options.exposure_compensation;
        }

        if options.debug {
            let stats = compute_stats(&data);
            eprintln!(
                "[RESEARCH] After exposure comp ({:.3}x) - min: {:.6}, max: {:.6}, mean: {:.6}",
                options.exposure_compensation, stats.0, stats.1, stats.2
            );
        }
    }

    // Stage 6: Tone curve
    if !options.skip_tone_curve && !options.no_clip {
        // Priority: tone_curve_override > film_preset > default
        let curve = if let Some(override_curve) = &options.tone_curve_override {
            override_curve.clone()
        } else if let Some(preset) = &options.film_preset {
            preset.tone_curve.clone()
        } else {
            crate::models::ToneCurveParams::default()
        };

        if options.debug {
            eprintln!(
                "[RESEARCH] Applying tone curve: type={}, strength={:.2}",
                curve.curve_type, curve.strength
            );
        }

        apply_tone_curve(&mut data, &curve);

        if options.debug {
            let stats = compute_stats(&data);
            eprintln!(
                "[RESEARCH] After tone curve - min: {:.6}, max: {:.6}, mean: {:.6}",
                stats.0, stats.1, stats.2
            );
        }
    }

    // Stage 7: Final range enforcement
    if !options.no_clip {
        enforce_working_range(&mut data);
    }

    if options.debug {
        let stats = compute_stats(&data);
        eprintln!(
            "[RESEARCH] Final output - min: {:.6}, max: {:.6}, mean: {:.6}",
            stats.0, stats.1, stats.2
        );
    }

    Ok(ProcessedImage {
        width,
        height,
        data,
        channels,
        export_as_grayscale,
    })
}

/// Apply film base white balance by dividing each pixel by base RGB values.
///
/// This normalizes the orange mask so that the film base becomes neutral gray.
/// After this step, the orange mask influence is removed proportionally across
/// all tonal values (as the mask affects all densities proportionally).
fn apply_film_base_white_balance(
    data: &mut [f32],
    base: &BaseEstimation,
    options: &ConvertOptions,
) -> Result<(), String> {
    let [base_r, base_g, base_b] = base.medians;

    // Ensure we don't divide by zero
    let norm_r = base_r.max(0.0001);
    let norm_g = base_g.max(0.0001);
    let norm_b = base_b.max(0.0001);

    // Divide each pixel by base to normalize the mask
    for pixel in data.chunks_exact_mut(3) {
        pixel[0] /= norm_r;
        pixel[1] /= norm_g;
        pixel[2] /= norm_b;
    }

    if options.debug {
        eprintln!(
            "[RESEARCH] Film base WB: dividing by [{:.4}, {:.4}, {:.4}]",
            norm_r, norm_g, norm_b
        );
        let stats = compute_stats(data);
        eprintln!(
            "[RESEARCH] After film base WB - min: {:.6}, max: {:.6}, mean: {:.6}",
            stats.0, stats.1, stats.2
        );
    }

    Ok(())
}

/// Get density balance from options, or compute it from neutral point/auto-detection.
fn get_or_compute_density_balance(
    data: &[f32],
    width: u32,
    height: u32,
    options: &ConvertOptions,
) -> Result<DensityBalance, String> {
    // Priority 1: Explicit density_balance in options
    if let Some(db) = &options.density_balance {
        return Ok(db.clone());
    }

    // Priority 2: Manual red/blue overrides
    if options.density_balance_red.is_some() || options.density_balance_blue.is_some() {
        return Ok(DensityBalance::manual(
            options.density_balance_red.unwrap_or(1.05),
            options.density_balance_blue.unwrap_or(0.90),
        ));
    }

    // Priority 3: Compute from neutral point sample (if provided)
    if let Some(ref neutral) = options.neutral_point {
        if neutral.roi.is_some() {
            // Sample the ROI to get neutral RGB
            let sampled = sample_neutral_roi(data, width, height, neutral)?;
            return Ok(DensityBalance::from_neutral_point(sampled));
        }
    }

    // Priority 4: Auto-detect neutral areas
    if let Some(neutral_rgb) = auto_detect_neutral_point(data, width, height) {
        if options.debug {
            eprintln!(
                "[RESEARCH] Auto-detected neutral point: [{:.4}, {:.4}, {:.4}]",
                neutral_rgb[0], neutral_rgb[1], neutral_rgb[2]
            );
        }
        return Ok(DensityBalance::from_neutral_point(neutral_rgb));
    }

    // Priority 5: Fall back to defaults
    if options.debug {
        eprintln!("[RESEARCH] Using default density balance [1.05, 1.0, 0.90]");
    }
    Ok(DensityBalance::default())
}

/// Sample a neutral point from a specified ROI
fn sample_neutral_roi(
    data: &[f32],
    width: u32,
    height: u32,
    neutral: &NeutralPointSample,
) -> Result<[f32; 3], String> {
    let (x, y, w, h) = neutral.roi.ok_or("Neutral point ROI not specified")?;

    if x + w > width || y + h > height {
        return Err(format!(
            "Neutral ROI ({},{},{},{}) exceeds image bounds ({}x{})",
            x, y, w, h, width, height
        ));
    }

    let mut sum_r = 0.0f64;
    let mut sum_g = 0.0f64;
    let mut sum_b = 0.0f64;
    let mut count = 0u32;

    for py in y..(y + h) {
        for px in x..(x + w) {
            let idx = ((py * width + px) * 3) as usize;
            sum_r += data[idx] as f64;
            sum_g += data[idx + 1] as f64;
            sum_b += data[idx + 2] as f64;
            count += 1;
        }
    }

    if count == 0 {
        return Err("Empty neutral point ROI".to_string());
    }

    let n = count as f64;
    Ok([(sum_r / n) as f32, (sum_g / n) as f32, (sum_b / n) as f32])
}

/// Auto-detect neutral areas in the image for density balance calculation.
///
/// Finds pixels where R approx G approx B (low saturation, mid-brightness) and
/// returns the median of those samples.
fn auto_detect_neutral_point(data: &[f32], _width: u32, _height: u32) -> Option<[f32; 3]> {
    let mut neutral_candidates: Vec<[f32; 3]> = Vec::new();

    for pixel in data.chunks_exact(3) {
        let [r, g, b] = [pixel[0], pixel[1], pixel[2]];
        let avg = (r + g + b) / 3.0;

        // Skip very dark or very bright pixels
        if !(0.15..=0.85).contains(&avg) {
            continue;
        }

        // Check for low saturation (neutral gray)
        let max_deviation = (r - avg).abs().max((g - avg).abs()).max((b - avg).abs());
        let saturation = max_deviation / avg.max(0.001);

        // Accept pixels with < 10% saturation as neutral candidates
        if saturation < 0.10 {
            neutral_candidates.push([r, g, b]);
        }
    }

    if neutral_candidates.len() < 100 {
        return None; // Not enough neutral samples
    }

    // Compute median neutral value
    let mid = neutral_candidates.len() / 2;
    let mut r_vals: Vec<f32> = neutral_candidates.iter().map(|p| p[0]).collect();
    let mut g_vals: Vec<f32> = neutral_candidates.iter().map(|p| p[1]).collect();
    let mut b_vals: Vec<f32> = neutral_candidates.iter().map(|p| p[2]).collect();

    // Use partial sort for efficiency
    r_vals.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
    g_vals.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
    b_vals.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());

    Some([r_vals[mid], g_vals[mid], b_vals[mid]])
}

/// Apply density balance using per-channel power functions.
///
/// This is THE CRITICAL STEP that aligns characteristic curves before inversion.
/// Each film layer has slightly different gamma, causing color crossover.
/// Density balance corrects this by applying: R^db_r, G^db_g, B^db_b
fn apply_density_balance(
    data: &mut [f32],
    balance: &DensityBalance,
    options: &ConvertOptions,
) -> Result<(), String> {
    let [exp_r, exp_g, exp_b] = balance.exponents;

    // Skip if all exponents are 1.0 (no change)
    if (exp_r - 1.0).abs() < 0.001 && (exp_g - 1.0).abs() < 0.001 && (exp_b - 1.0).abs() < 0.001 {
        if options.debug {
            eprintln!("[RESEARCH] Density balance skipped (all exponents approx 1.0)");
        }
        return Ok(());
    }

    for pixel in data.chunks_exact_mut(3) {
        // Ensure positive values for power function
        let r = pixel[0].max(0.0001);
        let g = pixel[1].max(0.0001);
        let b = pixel[2].max(0.0001);

        pixel[0] = r.powf(exp_r);
        pixel[1] = g.powf(exp_g);
        pixel[2] = b.powf(exp_b);
    }

    if options.debug {
        eprintln!(
            "[RESEARCH] Density balance applied: R^{:.3}, G^{:.3}, B^{:.3} (source: {:?})",
            exp_r, exp_g, exp_b, balance.source
        );
        let stats = compute_stats(data);
        eprintln!(
            "[RESEARCH] After density balance - min: {:.6}, max: {:.6}, mean: {:.6}",
            stats.0, stats.1, stats.2
        );
    }

    Ok(())
}

// ============================================================================
// End Research Pipeline Implementation
// ============================================================================

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
///
/// Uses simple clamping to working range to preserve highlight detail
#[inline(always)]
fn apply_color_matrix_to_pixel(pixel: &mut [f32], matrix: &[[f32; 3]; 3]) {
    let r = pixel[0];
    let g = pixel[1];
    let b = pixel[2];

    // Matrix multiplication: output = matrix * input
    // Clamp to working range (no soft-clip to preserve highlight detail)
    pixel[0] = clamp_to_working_range(matrix[0][0] * r + matrix[0][1] * g + matrix[0][2] * b);
    pixel[1] = clamp_to_working_range(matrix[1][0] * r + matrix[1][1] * g + matrix[1][2] * b);
    pixel[2] = clamp_to_working_range(matrix[2][0] * r + matrix[2][1] * g + matrix[2][2] * b);
}

/// Clamp all values into the working range to avoid clipped blacks or whites.
pub fn enforce_working_range(data: &mut [f32]) {
    for value in data.iter_mut() {
        *value = clamp_to_working_range(*value);
    }
}

#[inline]
fn clamp_to_working_range(value: f32) -> f32 {
    value.clamp(WORKING_RANGE_FLOOR, WORKING_RANGE_CEILING)
}

/// Compute min, max, and mean statistics for debug output
pub fn compute_stats(data: &[f32]) -> (f32, f32, f32) {
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

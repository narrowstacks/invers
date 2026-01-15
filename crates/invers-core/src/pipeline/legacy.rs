//! Legacy processing pipeline
//!
//! Original Invers implementation with multiple inversion modes.

use super::{
    apply_color_matrix, apply_scan_profile_fused, apply_tone_curve, clamp_to_working_range,
    compute_stats, enforce_working_range, invert_negative, ProcessedImage,
};
use crate::decoders::DecodedImage;
use crate::models::ConvertOptions;

#[cfg(feature = "gpu")]
use crate::gpu;

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
pub(crate) fn process_image_legacy(
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
    // Step 1: Get base estimation (by reference to avoid clone)
    let base_estimation = match &options.base_estimation {
        Some(base) => base,
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
    invert_negative(&mut data, base_estimation, channels, options)?;

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
    // OPTIMIZATION: Fused gamma + HSL into single pass for better cache locality
    if let Some(ref scan_profile) = options.scan_profile {
        let gamma = scan_profile.default_gamma;
        let hsl_adj = scan_profile.hsl_adjustments.as_ref();

        let has_gamma = gamma.is_some_and(|g| g != [1.0, 1.0, 1.0]);
        let has_hsl = hsl_adj.is_some_and(|h| h.has_adjustments());

        if has_gamma || has_hsl {
            // Fused pass: apply gamma then HSL to each pixel
            apply_scan_profile_fused(&mut data, gamma, hsl_adj);

            if options.debug {
                let stats = compute_stats(&data);
                if has_gamma {
                    let g = gamma.unwrap();
                    eprintln!(
                        "[DEBUG] After scan profile gamma [{:.2}, {:.2}, {:.2}] + HSL - min: {:.6}, max: {:.6}, mean: {:.6}",
                        g[0], g[1], g[2], stats.0, stats.1, stats.2
                    );
                } else {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_headroom_zero_to_floor() {
        let mut data = vec![0.0, 0.0, 0.0];
        apply_headroom(&mut data);

        for &val in &data {
            assert!(
                (val - 0.005).abs() < 0.001,
                "Zero should map to ~0.005, got {}",
                val
            );
        }
    }

    #[test]
    fn test_apply_headroom_one_to_ceiling() {
        let mut data = vec![1.0, 1.0, 1.0];
        apply_headroom(&mut data);

        for &val in &data {
            assert!(
                (val - 0.98).abs() < 0.001,
                "One should map to ~0.98, got {}",
                val
            );
        }
    }

    #[test]
    fn test_apply_headroom_midpoint() {
        let mut data = vec![0.5, 0.5, 0.5];
        apply_headroom(&mut data);

        // 0.005 + 0.5 * 0.975 = 0.4925
        for &val in &data {
            assert!(
                (val - 0.4925).abs() < 0.01,
                "Midpoint should map to ~0.4925, got {}",
                val
            );
        }
    }
}

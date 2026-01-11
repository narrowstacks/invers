//! Negative-to-positive inversion algorithms
//!
//! This module contains the core inversion functions that convert scanned
//! film negatives to positive images. Different inversion modes are provided
//! to handle various film types and achieve different aesthetic results.

use crate::models::{BaseEstimation, ConvertOptions, InversionMode, ShadowLiftMode};

/// Subtract base and invert to positive
///
/// This is the core inversion function that converts negative film to positive.
/// It supports multiple inversion modes for different film types and looks.
pub fn invert_negative(
    data: &mut [f32],
    base: &BaseEstimation,
    channels: u8,
    options: &ConvertOptions,
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
        InversionMode::Linear => {
            // Linear inversion: positive = (base - negative) / base
            for pixel in data.chunks_exact_mut(3) {
                pixel[0] = (base_r - pixel[0]) / base_r;
                pixel[1] = (base_g - pixel[1]) / base_g;
                pixel[2] = (base_b - pixel[2]) / base_b;
            }
        }
        InversionMode::Logarithmic => {
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
        InversionMode::DivideBlend => {
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
        InversionMode::MaskAware => {
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
            let mask_profile = base.mask_profile.clone().unwrap_or_default();

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
        InversionMode::BlackAndWhite => {
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
        ShadowLiftMode::Fixed => {
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
        ShadowLiftMode::Percentile => {
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
        ShadowLiftMode::None => {
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

/// Apply inversion to convert normalized negative to positive.
///
/// After dividing by base and applying density balance, values are normalized:
/// - Film base (brightest) -> 1.0
/// - Scene shadows (near base brightness) -> close to 1.0
/// - Scene highlights (darkest) -> much less than 1.0
///
/// The inversion maps this to positive space:
/// - Film base (1.0) -> 0.0 (black, representing scene shadows)
/// - Scene shadows (~0.9) -> small positive value (dark)
/// - Scene highlights (~0.3) -> large positive value (bright)
///
/// Formula: positive = 1.0 - normalized_negative
/// This is equivalent to (base - pixel) / base after the normalization step.
pub fn apply_reciprocal_inversion(
    data: &mut [f32],
    options: &ConvertOptions,
) -> Result<(), String> {
    for pixel in data.chunks_exact_mut(3) {
        // Inversion: positive = 1.0 - normalized
        // This correctly maps film base (1.0) to black and scene highlights (<1.0) to bright
        pixel[0] = 1.0 - pixel[0];
        pixel[1] = 1.0 - pixel[1];
        pixel[2] = 1.0 - pixel[2];
    }

    if options.debug {
        let stats = super::compute_stats(data);
        eprintln!(
            "[RESEARCH] After inversion - min: {:.6}, max: {:.6}, mean: {:.6}",
            stats.0, stats.1, stats.2
        );
    }

    Ok(())
}

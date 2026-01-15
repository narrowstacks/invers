//! Research-based processing pipeline
//!
//! Implements densitometry principles with density balance BEFORE inversion
//! to eliminate color crossover between shadows and highlights.

use super::{
    apply_reciprocal_inversion, apply_tone_curve, compute_stats, enforce_working_range,
    ProcessedImage,
};
use crate::decoders::DecodedImage;
use crate::models::{BaseEstimation, ConvertOptions, DensityBalance, NeutralPointSample};

#[cfg(feature = "gpu")]
use crate::gpu;

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
pub(crate) fn process_image_research(
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
    // Step 1: Get base estimation (by reference to avoid clone)
    let base_estimation = match &options.base_estimation {
        Some(base) => base,
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
    apply_film_base_white_balance(&mut data, base_estimation, options)?;

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
pub(crate) fn apply_film_base_white_balance(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::DensityBalanceSource;

    #[test]
    fn test_apply_film_base_white_balance() {
        let base = BaseEstimation {
            medians: [0.8, 0.6, 0.4],
            roi: None,
            noise_stats: None,
            auto_estimated: true,
            mask_profile: None,
        };

        let mut data = vec![0.4, 0.3, 0.2]; // One pixel
        let options = ConvertOptions::default();

        let result = apply_film_base_white_balance(&mut data, &base, &options);
        assert!(result.is_ok());

        // 0.4 / 0.8 = 0.5, 0.3 / 0.6 = 0.5, 0.2 / 0.4 = 0.5
        assert!(
            (data[0] - 0.5).abs() < 0.001,
            "R should be 0.5, got {}",
            data[0]
        );
        assert!(
            (data[1] - 0.5).abs() < 0.001,
            "G should be 0.5, got {}",
            data[1]
        );
        assert!(
            (data[2] - 0.5).abs() < 0.001,
            "B should be 0.5, got {}",
            data[2]
        );
    }

    #[test]
    fn test_apply_film_base_white_balance_avoids_divide_by_zero() {
        let base = BaseEstimation {
            medians: [0.0, 0.0, 0.0], // Zero base (should be clamped)
            roi: None,
            noise_stats: None,
            auto_estimated: true,
            mask_profile: None,
        };

        let mut data = vec![0.5, 0.5, 0.5];
        let options = ConvertOptions::default();

        let result = apply_film_base_white_balance(&mut data, &base, &options);
        assert!(result.is_ok());

        // Should not produce NaN or Inf
        for &val in &data {
            assert!(val.is_finite(), "Result should be finite: {}", val);
        }
    }

    #[test]
    fn test_apply_density_balance() {
        let balance = DensityBalance {
            exponents: [1.1, 1.0, 0.9],
            source: DensityBalanceSource::Manual,
        };

        let mut data = vec![0.5, 0.5, 0.5];
        let options = ConvertOptions::default();

        let result = apply_density_balance(&mut data, &balance, &options);
        assert!(result.is_ok());

        // 0.5^1.1 < 0.5, 0.5^1.0 = 0.5, 0.5^0.9 > 0.5
        assert!(data[0] < 0.5, "R with exp 1.1 should decrease: {}", data[0]);
        assert!(
            (data[1] - 0.5).abs() < 0.001,
            "G with exp 1.0 should stay: {}",
            data[1]
        );
        assert!(data[2] > 0.5, "B with exp 0.9 should increase: {}", data[2]);
    }

    #[test]
    fn test_apply_density_balance_skips_identity() {
        let balance = DensityBalance {
            exponents: [1.0, 1.0, 1.0], // All identity
            source: DensityBalanceSource::Default,
        };

        let mut data = vec![0.5, 0.5, 0.5];
        let original = data.clone();
        let options = ConvertOptions::default();

        let result = apply_density_balance(&mut data, &balance, &options);
        assert!(result.is_ok());

        // Should be unchanged
        for (i, (&orig, &new)) in original.iter().zip(data.iter()).enumerate() {
            assert!(
                (orig - new).abs() < 0.001,
                "Identity exponents should not change values at {}",
                i
            );
        }
    }

    #[test]
    fn test_auto_detect_neutral_point_finds_gray() {
        // Create data with neutral gray pixels
        let mut data = Vec::new();
        // Add neutral gray pixels (low saturation)
        for _ in 0..200 {
            data.extend_from_slice(&[0.5, 0.5, 0.5]);
        }
        // Add some colored pixels
        for _ in 0..100 {
            data.extend_from_slice(&[0.8, 0.2, 0.1]);
        }

        let result = auto_detect_neutral_point(&data, 100, 3);

        assert!(result.is_some(), "Should detect neutral point");
        let [r, g, b] = result.unwrap();
        assert!((r - 0.5).abs() < 0.1, "Neutral R should be ~0.5, got {}", r);
        assert!((g - 0.5).abs() < 0.1, "Neutral G should be ~0.5, got {}", g);
        assert!((b - 0.5).abs() < 0.1, "Neutral B should be ~0.5, got {}", b);
    }

    #[test]
    fn test_auto_detect_neutral_point_no_gray() {
        // Create data with only saturated colors
        let mut data = Vec::new();
        for _ in 0..100 {
            data.extend_from_slice(&[0.9, 0.1, 0.1]); // Saturated red
            data.extend_from_slice(&[0.1, 0.9, 0.1]); // Saturated green
            data.extend_from_slice(&[0.1, 0.1, 0.9]); // Saturated blue
        }

        let result = auto_detect_neutral_point(&data, 100, 3);
        assert!(
            result.is_none(),
            "Should not detect neutral point in saturated image"
        );
    }

    #[test]
    fn test_sample_neutral_roi() {
        // Create uniform gray data
        let data: Vec<f32> = vec![0.5; 300]; // 10x10 pixels
        let neutral = NeutralPointSample {
            roi: Some((2, 2, 3, 3)), // 3x3 region in center
            neutral_rgb: [0.5, 0.5, 0.5],
            auto_detected: false,
        };

        let result = sample_neutral_roi(&data, 10, 10, &neutral);
        assert!(result.is_ok());

        let [r, g, b] = result.unwrap();
        assert!((r - 0.5).abs() < 0.001);
        assert!((g - 0.5).abs() < 0.001);
        assert!((b - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_sample_neutral_roi_out_of_bounds() {
        let data: Vec<f32> = vec![0.5; 300];
        let neutral = NeutralPointSample {
            roi: Some((8, 8, 5, 5)), // Extends beyond 10x10 image
            neutral_rgb: [0.5, 0.5, 0.5],
            auto_detected: false,
        };

        let result = sample_neutral_roi(&data, 10, 10, &neutral);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("exceeds image bounds"));
    }
}

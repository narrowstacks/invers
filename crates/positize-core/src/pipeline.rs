//! Image processing pipeline
//!
//! Core pipeline for negative-to-positive conversion.

use crate::decoders::DecodedImage;
use crate::models::{BaseEstimation, ConvertOptions};

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

    // Step 2: Clone image data for processing (we'll modify it in-place)
    let mut data = image.data.clone();
    let width = image.width;
    let height = image.height;
    let channels = image.channels;

    if channels != 3 {
        return Err(format!("Pipeline requires 3-channel RGB, got {}", channels));
    }

    // Step 3: Base subtraction and inversion
    invert_negative(&mut data, &base_estimation, channels)?;

    if options.debug {
        let stats = compute_stats(&data);
        eprintln!("[DEBUG] After inversion - min: {:.6}, max: {:.6}, mean: {:.6}",
                  stats.0, stats.1, stats.2);
    }

    // Step 3.5: Apply exposure compensation if requested
    if (options.exposure_compensation - 1.0).abs() > 0.001 {
        for value in data.iter_mut() {
            *value = (*value * options.exposure_compensation).clamp(0.0, 1.0);
        }

        if options.debug {
            let stats = compute_stats(&data);
            eprintln!("[DEBUG] After exposure {:.2}x - min: {:.6}, max: {:.6}, mean: {:.6}",
                      options.exposure_compensation, stats.0, stats.1, stats.2);
        }
    }

    // Step 4: Apply tone curve (unless skipped)
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
            eprintln!("[DEBUG] After tone curve - min: {:.6}, max: {:.6}, mean: {:.6}",
                      stats.0, stats.1, stats.2);
        }
    } else if options.debug {
        eprintln!("[DEBUG] Tone curve skipped");
    }

    // Step 5: Apply color correction matrix (unless skipped)
    if !options.skip_color_matrix {
        if let Some(preset) = &options.film_preset {
            apply_color_matrix(&mut data, &preset.color_matrix, channels);

            if options.debug {
                let stats = compute_stats(&data);
                eprintln!("[DEBUG] After color matrix - min: {:.6}, max: {:.6}, mean: {:.6}",
                          stats.0, stats.1, stats.2);
            }
        }
    } else if options.debug {
        eprintln!("[DEBUG] Color matrix skipped");
    }

    // Step 6: Colorspace transform (for now, keep in same space)
    // TODO: Implement colorspace transforms in M3

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
    // If no ROI provided, use auto-estimation heuristic
    let roi_rect = match roi {
        Some(r) => r,
        None => estimate_base_roi(image)?,
    };

    let (x, y, width, height) = roi_rect;

    // Validate ROI bounds
    if x + width > image.width || y + height > image.height {
        return Err(format!(
            "ROI out of bounds: ({}, {}, {}, {}) exceeds image size {}x{}",
            x, y, width, height, image.width, image.height
        ));
    }

    if width == 0 || height == 0 {
        return Err("ROI width and height must be greater than 0".to_string());
    }

    // Extract ROI pixels
    let roi_pixels = extract_roi_pixels(image, x, y, width, height);

    // Compute per-channel medians
    let medians = compute_channel_medians(&roi_pixels);

    // Calculate noise statistics (standard deviation per channel)
    let noise_stats = compute_noise_stats(&roi_pixels, &medians);

    Ok(BaseEstimation {
        roi: Some(roi_rect),
        medians,
        noise_stats: Some(noise_stats),
        auto_estimated: roi.is_none(),
    })
}

/// Extract pixels from a region of interest
fn extract_roi_pixels(
    image: &DecodedImage,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) -> Vec<[f32; 3]> {
    let mut pixels = Vec::with_capacity((width * height) as usize);

    for row in y..(y + height) {
        for col in x..(x + width) {
            let pixel_idx = ((row * image.width + col) * 3) as usize;
            if pixel_idx + 2 < image.data.len() {
                let r = image.data[pixel_idx];
                let g = image.data[pixel_idx + 1];
                let b = image.data[pixel_idx + 2];
                pixels.push([r, g, b]);
            }
        }
    }

    pixels
}

/// Compute per-channel medians
fn compute_channel_medians(pixels: &[[f32; 3]]) -> [f32; 3] {
    if pixels.is_empty() {
        return [0.0, 0.0, 0.0];
    }

    let mut r_values: Vec<f32> = pixels.iter().map(|p| p[0]).collect();
    let mut g_values: Vec<f32> = pixels.iter().map(|p| p[1]).collect();
    let mut b_values: Vec<f32> = pixels.iter().map(|p| p[2]).collect();

    [
        compute_median(&mut r_values),
        compute_median(&mut g_values),
        compute_median(&mut b_values),
    ]
}

/// Compute median of a slice
fn compute_median(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }

    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let len = values.len();
    if len % 2 == 0 {
        // Even length: average of two middle values
        (values[len / 2 - 1] + values[len / 2]) / 2.0
    } else {
        // Odd length: middle value
        values[len / 2]
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

/// Estimate film base ROI automatically using heuristics
/// For color negatives, looks for high-value (bright orange) regions
/// For B&W, looks for high-value regions
fn estimate_base_roi(image: &DecodedImage) -> Result<(u32, u32, u32, u32), String> {
    // Simple heuristic: sample the border regions (top, bottom, left, right)
    // and find the brightest region, which typically corresponds to film base

    let border_width = (image.width / 20).max(50).min(200);
    let border_height = (image.height / 20).max(50).min(200);

    // Sample top border
    let top_brightness = sample_region_brightness(image, 0, 0, image.width, border_height);

    // Sample bottom border
    let bottom_brightness = sample_region_brightness(
        image,
        0,
        image.height.saturating_sub(border_height),
        image.width,
        border_height,
    );

    // Sample left border
    let left_brightness = sample_region_brightness(image, 0, 0, border_width, image.height);

    // Sample right border
    let right_brightness = sample_region_brightness(
        image,
        image.width.saturating_sub(border_width),
        0,
        border_width,
        image.height,
    );

    // Find brightest region
    let regions = [
        (top_brightness, (border_width, 0, image.width - 2 * border_width, border_height)),
        (bottom_brightness, (border_width, image.height.saturating_sub(border_height), image.width - 2 * border_width, border_height)),
        (left_brightness, (0, border_height, border_width, image.height - 2 * border_height)),
        (right_brightness, (image.width.saturating_sub(border_width), border_height, border_width, image.height - 2 * border_height)),
    ];

    let (_, roi) = regions
        .iter()
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .ok_or("Failed to determine auto ROI".to_string())?;

    Ok(*roi)
}

/// Sample the average brightness of a region
fn sample_region_brightness(image: &DecodedImage, x: u32, y: u32, width: u32, height: u32) -> f32 {
    let x_end = (x + width).min(image.width);
    let y_end = (y + height).min(image.height);

    let mut sum = 0.0;
    let mut count = 0;

    for row in y..y_end {
        for col in x..x_end {
            let pixel_idx = ((row * image.width + col) * 3) as usize;
            if pixel_idx + 2 < image.data.len() {
                // Average RGB values for brightness
                sum += image.data[pixel_idx] + image.data[pixel_idx + 1] + image.data[pixel_idx + 2];
                count += 3;
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
) -> Result<(), String> {
    if channels != 3 {
        return Err(format!("Expected 3 channels, got {}", channels));
    }

    if data.len() % 3 != 0 {
        return Err(format!(
            "Data length {} is not divisible by 3 (RGB)",
            data.len()
        ));
    }

    let base_r = base.medians[0];
    let base_g = base.medians[1];
    let base_b = base.medians[2];

    // Invert using base-normalized subtraction with shadow lift
    // Per-channel normalization handles orange mask correctly
    // Shadow lift prevents pure black and preserves shadow detail
    //
    // Formula: ((base - pixel) / base) + lift
    // - Clear film (pixel=0) → (base / base) + lift = 1.0 + lift (bright, but not max)
    // - Film base (pixel=base) → (0 / base) + lift = lift (near black but preserves detail)
    // - Dense film (pixel>base) → negative + lift (may be < lift)

    let base_r = base_r.max(0.0001);
    let base_g = base_g.max(0.0001);
    let base_b = base_b.max(0.0001);

    //Scale and shift to preserve information in a smaller range
    // Instead of using full [0, 1], map to approximately [0.15, 0.85]
    // This prevents both shadow crushing (min > 0) and highlight clipping (max < 1)
    let scale = 0.7; // Scale factor to prevent highlight clipping
    let lift = 0.15; // Shadow lift to preserve shadow detail

    for pixel in data.chunks_exact_mut(3) {
        // Compute relative density, scale down, and lift shadows
        // Formula: (base - pixel) / base * scale + lift
        let r = ((base_r - pixel[0]) / base_r) * scale + lift;
        let g = ((base_g - pixel[1]) / base_g) * scale + lift;
        let b = ((base_b - pixel[2]) / base_b) * scale + lift;

        pixel[0] = r.clamp(0.0, 1.0);
        pixel[1] = g.clamp(0.0, 1.0);
        pixel[2] = b.clamp(0.0, 1.0);
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

    result.clamp(0.0, 1.0)
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

        pixel[0] = (matrix[0][0] * r + matrix[0][1] * g + matrix[0][2] * b).clamp(0.0, 1.0);
        pixel[1] = (matrix[1][0] * r + matrix[1][1] * g + matrix[1][2] * b).clamp(0.0, 1.0);
        pixel[2] = (matrix[2][0] * r + matrix[2][1] * g + matrix[2][2] * b).clamp(0.0, 1.0);
    }
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

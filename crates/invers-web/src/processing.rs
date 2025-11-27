//! Processing bridge to invers-core
//!
//! This module provides the interface between the web UI and the invers-core
//! processing pipeline. It handles image loading, preview generation, and export.

use crate::embedded_presets;
use crate::state::{AppState, MAX_IMAGE_MEGAPIXELS, MAX_PREVIEW_SIZE};
use invers_core::decoders::{decode_image_from_bytes, DecodedImage, ImageFormat};
use invers_core::exporters::export_tiff16_to_bytes;
use invers_core::models::{
    BaseEstimation, BitDepthPolicy, ConvertOptions, OutputFormat, ToneCurveParams,
};
use invers_core::pipeline::{estimate_base, process_image, ProcessedImage};
use std::path::PathBuf;

/// Load an image from bytes
pub fn load_image(data: &[u8], filename: &str) -> Result<DecodedImage, String> {
    let format = ImageFormat::from_filename(filename)
        .ok_or_else(|| format!("Unsupported file format: {}", filename))?;

    let image = decode_image_from_bytes(data, format)?;

    // Check image size limit
    let megapixels = (image.width as u64 * image.height as u64) / 1_000_000;
    if megapixels > MAX_IMAGE_MEGAPIXELS as u64 {
        return Err(format!(
            "Image too large: {} MP (max {} MP)",
            megapixels, MAX_IMAGE_MEGAPIXELS
        ));
    }

    Ok(image)
}

/// Create a downsampled preview image
pub fn create_preview(image: &DecodedImage) -> DecodedImage {
    let max_dim = image.width.max(image.height);

    if max_dim <= MAX_PREVIEW_SIZE {
        // Image is already small enough, return a clone
        return image.clone();
    }

    // Calculate new dimensions maintaining aspect ratio
    let scale = MAX_PREVIEW_SIZE as f32 / max_dim as f32;
    let new_width = (image.width as f32 * scale).round() as u32;
    let new_height = (image.height as f32 * scale).round() as u32;

    // Simple bilinear downsampling
    let mut preview_data = Vec::with_capacity((new_width * new_height * 3) as usize);

    for y in 0..new_height {
        for x in 0..new_width {
            // Map to source coordinates
            let src_x = (x as f32 / new_width as f32) * (image.width - 1) as f32;
            let src_y = (y as f32 / new_height as f32) * (image.height - 1) as f32;

            // Bilinear interpolation
            let x0 = src_x.floor() as u32;
            let y0 = src_y.floor() as u32;
            let x1 = (x0 + 1).min(image.width - 1);
            let y1 = (y0 + 1).min(image.height - 1);

            let fx = src_x - x0 as f32;
            let fy = src_y - y0 as f32;

            for c in 0..3 {
                let p00 = get_pixel(&image.data, image.width, x0, y0, c);
                let p10 = get_pixel(&image.data, image.width, x1, y0, c);
                let p01 = get_pixel(&image.data, image.width, x0, y1, c);
                let p11 = get_pixel(&image.data, image.width, x1, y1, c);

                let value = p00 * (1.0 - fx) * (1.0 - fy)
                    + p10 * fx * (1.0 - fy)
                    + p01 * (1.0 - fx) * fy
                    + p11 * fx * fy;

                preview_data.push(value);
            }
        }
    }

    DecodedImage {
        width: new_width,
        height: new_height,
        data: preview_data,
        channels: 3,
        black_level: image.black_level,
        white_level: image.white_level,
        color_matrix: image.color_matrix,
    }
}

/// Get a pixel value from linear data
fn get_pixel(data: &[f32], width: u32, x: u32, y: u32, channel: usize) -> f32 {
    let idx = ((y * width + x) * 3) as usize + channel;
    data.get(idx).copied().unwrap_or(0.0)
}

/// Estimate film base from an image
pub fn estimate_film_base(image: &DecodedImage) -> Result<BaseEstimation, String> {
    estimate_base(image, None, None, None)
}

/// Process a preview image with the current state settings
pub fn process_preview(state: &AppState) -> Result<ProcessedImage, String> {
    let preview = state
        .preview_image
        .get_clone()
        .get()
        .ok_or("No preview image loaded")?;

    let options = build_convert_options(state);
    process_image(preview, &options)
}

/// Process the full-resolution image for export
pub fn process_full_resolution(state: &AppState) -> Result<ProcessedImage, String> {
    let image = state
        .original_image
        .get_clone()
        .get()
        .ok_or("No image loaded")?;

    let options = build_convert_options(state);
    process_image(image, &options)
}

/// Build ConvertOptions from the current state
fn build_convert_options(state: &AppState) -> ConvertOptions {
    // Get film preset if selected
    let preset_slug = state.selected_preset.get_clone();
    let film_preset = embedded_presets::get_preset(&preset_slug).cloned();

    // Build tone curve params
    let tone_curve = if let Some(ref preset) = film_preset {
        ToneCurveParams {
            curve_type: preset.tone_curve.curve_type.clone(),
            strength: state.tone_curve_strength.get(),
            toe_strength: preset.tone_curve.toe_strength,
            shoulder_strength: preset.tone_curve.shoulder_strength,
            toe_length: preset.tone_curve.toe_length,
            shoulder_start: preset.tone_curve.shoulder_start,
            params: preset.tone_curve.params.clone(),
        }
    } else {
        ToneCurveParams {
            curve_type: "s-curve".to_string(),
            strength: state.tone_curve_strength.get(),
            ..Default::default()
        }
    };

    // Build color matrix from state, applying preset if not skipped
    let color_matrix = if state.skip_color_matrix.get() {
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    } else {
        state.color_matrix.get()
    };

    // Create film preset with current settings
    let preset_with_settings = film_preset.map(|mut p| {
        p.base_offsets = [
            state.base_offset_r.get(),
            state.base_offset_g.get(),
            state.base_offset_b.get(),
        ];
        p.color_matrix = color_matrix;
        p.tone_curve = tone_curve.clone();
        p
    });

    // Build base estimation from state
    let base_estimation = BaseEstimation {
        roi: None,
        medians: [state.base_r.get(), state.base_g.get(), state.base_b.get()],
        noise_stats: None,
        auto_estimated: true,
    };

    ConvertOptions {
        input_paths: vec![],
        output_dir: PathBuf::new(),
        output_format: OutputFormat::Tiff16,
        working_colorspace: "linear-rec2020".to_string(),
        bit_depth_policy: BitDepthPolicy::Force16Bit,
        film_preset: preset_with_settings,
        scan_profile: None,
        base_estimation: Some(base_estimation),
        num_threads: None,
        skip_tone_curve: state.skip_tone_curve.get(),
        skip_color_matrix: state.skip_color_matrix.get(),
        exposure_compensation: state.exposure.get(),
        debug: false,
        enable_auto_levels: state.enable_auto_levels.get(),
        auto_levels_clip_percent: state.auto_levels_clip.get(),
        preserve_headroom: true,
        enable_auto_color: state.enable_auto_color.get(),
        auto_color_strength: state.auto_color_strength.get(),
        auto_color_min_gain: 0.7,
        auto_color_max_gain: 1.3,
        base_brightest_percent: 5.0,
        base_sampling_mode: Default::default(),
        inversion_mode: Default::default(),
        shadow_lift_mode: Default::default(),
        shadow_lift_value: 0.02,
        highlight_compression: 1.0,
        enable_auto_exposure: state.enable_auto_exposure.get(),
        auto_exposure_target_median: 0.25,
        auto_exposure_strength: 1.0,
        auto_exposure_min_gain: 0.6,
        auto_exposure_max_gain: 1.4,
    }
}

/// Export processed image to TIFF bytes
pub fn export_to_tiff(processed: &ProcessedImage) -> Result<Vec<u8>, String> {
    export_tiff16_to_bytes(processed, None)
}

/// Convert linear RGB to sRGB for display (gamma correction)
pub fn linear_to_srgb(linear: f32) -> f32 {
    if linear <= 0.0031308 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
}

/// Convert processed image to RGBA bytes for canvas display
pub fn processed_to_rgba(processed: &ProcessedImage) -> Vec<u8> {
    let pixel_count = (processed.width * processed.height) as usize;
    let mut rgba = Vec::with_capacity(pixel_count * 4);

    for i in 0..pixel_count {
        let r = processed.data[i * 3];
        let g = processed.data[i * 3 + 1];
        let b = processed.data[i * 3 + 2];

        // Apply gamma correction for display
        let r_srgb = (linear_to_srgb(r.clamp(0.0, 1.0)) * 255.0).round() as u8;
        let g_srgb = (linear_to_srgb(g.clamp(0.0, 1.0)) * 255.0).round() as u8;
        let b_srgb = (linear_to_srgb(b.clamp(0.0, 1.0)) * 255.0).round() as u8;

        rgba.push(r_srgb);
        rgba.push(g_srgb);
        rgba.push(b_srgb);
        rgba.push(255); // Alpha
    }

    rgba
}

/// Sample a region from decoded image (for eyedropper)
pub fn sample_region(
    image: &DecodedImage,
    x: u32,
    y: u32,
    radius: u32,
) -> Option<(f32, f32, f32)> {
    let mut sum_r = 0.0;
    let mut sum_g = 0.0;
    let mut sum_b = 0.0;
    let mut count = 0;

    let x_start = x.saturating_sub(radius);
    let y_start = y.saturating_sub(radius);
    let x_end = (x + radius + 1).min(image.width);
    let y_end = (y + radius + 1).min(image.height);

    for sy in y_start..y_end {
        for sx in x_start..x_end {
            let idx = ((sy * image.width + sx) * 3) as usize;
            if idx + 2 < image.data.len() {
                sum_r += image.data[idx];
                sum_g += image.data[idx + 1];
                sum_b += image.data[idx + 2];
                count += 1;
            }
        }
    }

    if count > 0 {
        Some((
            sum_r / count as f32,
            sum_g / count as f32,
            sum_b / count as f32,
        ))
    } else {
        None
    }
}

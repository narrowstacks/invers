//! CB (Curves-Based) GPU pipeline implementation.

use super::dispatch::download_subsampled;
use super::ops_batched::{
    apply_cb_inversion_batched, apply_cb_layers_batched, apply_gains_batched,
};
use super::CommandBatch;
use crate::cb_pipeline::{
    analyze_histogram, analyze_wb_points, calculate_wb_gamma, calculate_wb_offsets,
    calculate_wb_preset_offsets,
};
use crate::decoders::DecodedImage;
use crate::gpu::buffers::{CbLayerParams, GpuImage};
use crate::gpu::context::{GpuContext, GpuError};
use crate::models::{AutoWbMode, CbLayerOrder, CbWbMethod, CbWbPreset, ConvertOptions};
use crate::pipeline::ProcessedImage;

/// Process an image using the CB pipeline on the GPU with batched command submission.
///
/// The CB pipeline uses fewer sync points than the legacy pipeline since it has
/// less CPU analysis. Sync points are:
/// 1. After CB inversion (before downloading subsampled data for WB analysis)
/// 2. Final sync at end (implicit via download)
pub fn process_image_cb_gpu(
    decoded: &DecodedImage,
    options: &ConvertOptions,
) -> Result<ProcessedImage, GpuError> {
    if decoded.channels != 3 {
        return Err(GpuError::Other(format!(
            "CB pipeline requires 3-channel RGB, got {}",
            decoded.channels
        )));
    }

    let cb = options.cb_options.clone().unwrap_or_default();

    // Apply film base WB on CPU (if available) and analyze histogram before upload.
    let mut cpu_data = decoded.data.clone();
    if let Some(base_estimation) = options.base_estimation.as_ref() {
        crate::pipeline::apply_film_base_white_balance(&mut cpu_data, base_estimation, options)
            .map_err(GpuError::Other)?;
    }

    let analysis = analyze_histogram(
        &cpu_data,
        decoded.channels,
        cb.white_threshold,
        cb.black_threshold,
    );

    // Initialize GPU context and upload image.
    let ctx = GpuContext::new()?;
    let gpu_image = GpuImage::upload(
        ctx.device.clone(),
        ctx.queue.clone(),
        &cpu_data,
        decoded.width,
        decoded.height,
        decoded.channels as u32,
    )?;

    let pixel_count = gpu_image.pixel_count();

    // Create command batch for accumulating GPU operations
    let mut batch = CommandBatch::new(&ctx, "cb_pipeline_batch");

    // ========================================================================
    // Step 1: CB inversion using per-channel curves (batched)
    // ========================================================================
    apply_cb_inversion_batched(&mut batch, &ctx, &gpu_image, &analysis, true, pixel_count)?;

    // ========================================================================
    // SYNC POINT: WB analysis needs subsampled readback
    // ========================================================================
    // Use subsampled download for WB analysis (64x smaller transfer for stride=8)
    const WB_SUBSAMPLE_STRIDE: u32 = 8;
    let needs_wb_analysis = cb.wb_preset != CbWbPreset::None || options.enable_auto_wb;

    if needs_wb_analysis {
        // Flush pending commands before downloading subsampled data
        batch.flush();

        if cb.wb_preset != CbWbPreset::None {
            let subsampled = download_subsampled(&ctx, &gpu_image, WB_SUBSAMPLE_STRIDE)?;
            let wb_points = analyze_wb_points(&subsampled, decoded.channels);
            let offsets = calculate_wb_preset_offsets(cb.wb_preset, &wb_points, cb.film_character);
            let strength = options.auto_wb_strength;
            let scaled_offsets = [
                offsets[0] / 255.0 * strength,
                offsets[1] / 255.0 * strength,
                offsets[2] / 255.0 * strength,
            ];
            apply_gains_batched(
                &mut batch,
                &ctx,
                &gpu_image,
                [1.0, 1.0, 1.0],
                [-scaled_offsets[0], -scaled_offsets[1], -scaled_offsets[2]],
                pixel_count,
            )?;
        } else if options.enable_auto_wb {
            let subsampled = download_subsampled(&ctx, &gpu_image, WB_SUBSAMPLE_STRIDE)?;

            let multipliers = match options.auto_wb_mode {
                AutoWbMode::GrayPixel => crate::auto_adjust::compute_wb_multipliers_gray_pixel(
                    &subsampled,
                    decoded.channels,
                    options.auto_wb_strength,
                ),
                AutoWbMode::Average => crate::auto_adjust::compute_wb_multipliers_avg(
                    &subsampled,
                    decoded.channels,
                    options.auto_wb_strength,
                ),
                AutoWbMode::Percentile => crate::auto_adjust::compute_wb_multipliers_percentile(
                    &subsampled,
                    decoded.channels,
                    options.auto_wb_strength,
                    98.0,
                ),
            };

            // Apply multipliers on GPU (gains with zero offsets)
            apply_gains_batched(
                &mut batch,
                &ctx,
                &gpu_image,
                multipliers,
                [0.0, 0.0, 0.0],
                pixel_count,
            )?;
        }
    }

    // ========================================================================
    // Step 3: Apply CB layers (WB temp/tint, color gamma, tonal adjustments, toning)
    // ========================================================================
    let brightness_gamma = 1.0 / (1.0 + cb.brightness * 0.02);
    let exposure_factor = 1.0 / (1.0 + cb.exposure * 0.02);
    let color_offsets = [
        1.0 - cb.cyan * 0.01,
        1.0 - cb.tint * 0.01,
        1.0 - cb.temp * 0.01,
    ];

    let wb_offsets = calculate_wb_offsets(cb.wb_temp, cb.wb_tint, cb.wb_tonality);
    let wb_gamma = calculate_wb_gamma(cb.wb_temp, cb.wb_tint, cb.wb_tonality);

    let wb_method = match cb.wb_method {
        CbWbMethod::LinearFixed => 0,
        CbWbMethod::LinearDynamic => 1,
        CbWbMethod::ShadowWeighted => 2,
        CbWbMethod::HighlightWeighted => 3,
        CbWbMethod::MidtoneWeighted => 4,
    };

    let layer_order = match cb.layer_order {
        CbLayerOrder::ColorFirst => 0,
        CbLayerOrder::TonesFirst => 1,
    };

    let mut apply_flags = 0u32;
    if cb.wb_temp != 0.0 || cb.wb_tint != 0.0 {
        apply_flags |= 1;
    }
    if cb.shadow_cyan != 0.0 || cb.shadow_tint != 0.0 || cb.shadow_temp != 0.0 {
        apply_flags |= 1 << 1;
    }
    if cb.highlight_cyan != 0.0 || cb.highlight_tint != 0.0 || cb.highlight_temp != 0.0 {
        apply_flags |= 1 << 2;
    }

    let layer_params = CbLayerParams {
        wb_offsets: [wb_offsets[0], wb_offsets[1], wb_offsets[2], 0.0],
        wb_gamma: [wb_gamma[0], wb_gamma[1], wb_gamma[2], 0.0],
        color_offsets: [color_offsets[0], color_offsets[1], color_offsets[2], 0.0],
        tonal_0: [
            exposure_factor,
            brightness_gamma,
            cb.contrast,
            cb.highlights,
        ],
        tonal_1: [cb.shadows, cb.blacks, cb.whites, cb.shadow_range],
        tonal_2: [cb.highlight_range, 0.0, 0.0, 0.0],
        shadow_colors: [cb.shadow_cyan, cb.shadow_tint, cb.shadow_temp, 0.0],
        highlight_colors: [cb.highlight_cyan, cb.highlight_tint, cb.highlight_temp, 0.0],
        flags: [wb_method, layer_order, pixel_count, apply_flags],
    };

    apply_cb_layers_batched(&mut batch, &ctx, &gpu_image, &layer_params, pixel_count)?;

    // ========================================================================
    // FINAL SYNC: Submit all remaining commands before download
    // ========================================================================
    batch.finish();

    // Download final output.
    let result_data = gpu_image.download()?;

    let export_as_grayscale = decoded.source_is_grayscale || decoded.is_monochrome;

    Ok(ProcessedImage {
        width: decoded.width,
        height: decoded.height,
        data: result_data,
        channels: decoded.channels,
        export_as_grayscale,
    })
}

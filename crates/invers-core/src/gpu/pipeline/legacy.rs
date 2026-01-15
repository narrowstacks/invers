//! Legacy GPU pipeline implementation.

use super::analysis::{
    compute_auto_color_gains_fullimage, compute_auto_levels_gains, compute_exposure_gain_cpu,
    compute_wb_gains_cpu,
};
use super::dispatch::estimate_base_cpu;
use super::histogram::{accumulate_histogram, clear_histogram};
use super::ops_batched::{
    apply_base_offsets_batched, apply_color_matrix_batched, apply_exposure_batched,
    apply_gains_batched, apply_highlight_compression_batched, apply_hsl_adjustments_batched,
    apply_inversion_batched, apply_shadow_lift_batched, apply_tone_curve_batched,
    clamp_working_range_batched,
};
use super::CommandBatch;
use crate::decoders::DecodedImage;
use crate::gpu::buffers::{GpuHistogram, GpuImage};
use crate::gpu::context::{GpuContext, GpuError};
use crate::models::{ConvertOptions, ShadowLiftMode};
use crate::pipeline::ProcessedImage;

/// Process an image on the GPU.
pub fn process_image_gpu(
    decoded: &DecodedImage,
    options: &ConvertOptions,
) -> Result<ProcessedImage, GpuError> {
    // Initialize GPU context
    let ctx = GpuContext::new()?;

    // Upload image to GPU
    let mut gpu_image = GpuImage::upload(
        ctx.device.clone(),
        ctx.queue.clone(),
        &decoded.data,
        decoded.width,
        decoded.height,
        decoded.channels as u32,
    )?;

    // Create histogram buffers
    let histogram = GpuHistogram::new(ctx.device.clone(), ctx.queue.clone());

    // Execute processing pipeline
    execute_pipeline(&ctx, &mut gpu_image, &histogram, decoded, options)?;

    // Download result
    let result_data = gpu_image.download()?;

    // Track if we should export as grayscale
    let export_as_grayscale = decoded.source_is_grayscale || decoded.is_monochrome;

    Ok(ProcessedImage {
        width: decoded.width,
        height: decoded.height,
        data: result_data,
        channels: decoded.channels,
        export_as_grayscale,
    })
}

/// Execute the full processing pipeline on GPU with batched command submission.
///
/// This is optimized to batch GPU commands and only synchronize when data needs
/// to be read back (for histogram analysis, auto-WB, auto-color, auto-exposure).
/// This can provide 2-4x speedup compared to syncing after every operation.
fn execute_pipeline(
    ctx: &GpuContext,
    image: &mut GpuImage,
    histogram: &GpuHistogram,
    decoded: &DecodedImage,
    options: &ConvertOptions,
) -> Result<(), GpuError> {
    let pixel_count = image.pixel_count();

    // Stage 1: Base estimation (done on CPU, we just need the results)
    let base = options
        .base_estimation
        .clone()
        .unwrap_or_else(|| estimate_base_cpu(decoded, options));

    // Create command batch for accumulating GPU operations
    let mut batch = CommandBatch::new(ctx, "pipeline_batch");

    // ========================================================================
    // BATCH 1: Inversion + Shadow lift + Highlight compression
    // These operations don't need data readback, so we can batch them.
    // ========================================================================

    // Stage 2: Inversion (always applied for negatives)
    apply_inversion_batched(
        &mut batch,
        ctx,
        image,
        &base,
        &options.inversion_mode,
        pixel_count,
    )?;

    // Stage 3: Shadow lift (if configured)
    if options.shadow_lift_mode != ShadowLiftMode::None {
        apply_shadow_lift_batched(&mut batch, ctx, image, options, pixel_count)?;
    }

    // Stage 4: Highlight compression (if configured)
    if options.highlight_compression < 1.0 {
        apply_highlight_compression_batched(
            &mut batch,
            ctx,
            image,
            0.9, // Default threshold
            options.highlight_compression,
            pixel_count,
        )?;
    }

    // ========================================================================
    // SYNC POINT: Auto-levels needs histogram readback
    // ========================================================================
    if options.enable_auto_levels {
        // Flush pending commands before histogram operations
        batch.flush();

        // Build histogram on GPU (uses its own sync)
        clear_histogram(ctx, histogram)?;
        accumulate_histogram(ctx, image, histogram)?;

        // Download histogram for percentile computation (small data transfer)
        let [hist_r, hist_g, hist_b] = histogram.download()?;

        // Compute gains on CPU
        let (gains, offsets) =
            compute_auto_levels_gains(&hist_r, &hist_g, &hist_b, options.auto_levels_clip_percent);

        // Apply gains on GPU (into the batch)
        apply_gains_batched(&mut batch, ctx, image, gains, offsets, pixel_count)?;
    }

    // Stage 6: Film preset base offsets (no sync needed)
    if let Some(ref preset) = options.film_preset {
        if preset.base_offsets != [0.0, 0.0, 0.0] {
            apply_base_offsets_batched(&mut batch, ctx, image, preset.base_offsets, pixel_count)?;
        }
    }

    // ========================================================================
    // SYNC POINT: Auto-WB needs image readback
    // ========================================================================
    if options.enable_auto_wb {
        batch.flush(); // Sync before download

        let wb_gains = compute_wb_gains_cpu(image, ctx)?;
        let strength = options.auto_wb_strength;
        let adjusted_gains = [
            1.0 + strength * (wb_gains[0] - 1.0),
            1.0 + strength * (wb_gains[1] - 1.0),
            1.0 + strength * (wb_gains[2] - 1.0),
        ];
        apply_gains_batched(
            &mut batch,
            ctx,
            image,
            adjusted_gains,
            [0.0, 0.0, 0.0],
            pixel_count,
        )?;
    }

    // ========================================================================
    // SYNC POINT: Auto-color needs image readback
    // ========================================================================
    if options.enable_auto_color {
        batch.flush(); // Sync before download

        let color_gains = compute_auto_color_gains_fullimage(image, ctx, options)?;
        apply_gains_batched(
            &mut batch,
            ctx,
            image,
            color_gains,
            [0.0, 0.0, 0.0],
            pixel_count,
        )?;
    }

    // ========================================================================
    // SYNC POINT: Auto-exposure needs image readback
    // ========================================================================
    if options.enable_auto_exposure {
        batch.flush(); // Sync before download

        let exposure_gain = compute_exposure_gain_cpu(image, ctx, options)?;
        let strength = options.auto_exposure_strength;
        let adjusted_gain = 1.0 + strength * (exposure_gain - 1.0);
        apply_exposure_batched(&mut batch, ctx, image, adjusted_gain, 1.0, pixel_count)?;
    }

    // ========================================================================
    // BATCH 2: Remaining operations (no sync needed until final download)
    // ========================================================================

    // Stage 10: Manual exposure compensation
    if options.exposure_compensation != 1.0 {
        apply_exposure_batched(
            &mut batch,
            ctx,
            image,
            options.exposure_compensation,
            1.0,
            pixel_count,
        )?;
    }

    // Stage 12: Color matrix (if film preset has one)
    if !options.skip_color_matrix {
        if let Some(ref preset) = options.film_preset {
            let matrix = preset.color_matrix;
            let is_identity = (matrix[0][0] - 1.0).abs() < 0.001
                && (matrix[1][1] - 1.0).abs() < 0.001
                && (matrix[2][2] - 1.0).abs() < 0.001
                && matrix[0][1].abs() < 0.001
                && matrix[0][2].abs() < 0.001
                && matrix[1][0].abs() < 0.001
                && matrix[1][2].abs() < 0.001
                && matrix[2][0].abs() < 0.001
                && matrix[2][1].abs() < 0.001;

            if !is_identity {
                apply_color_matrix_batched(&mut batch, ctx, image, &matrix, pixel_count)?;
            }
        }
    }

    // Stage 13: Tone curve
    if !options.skip_tone_curve {
        apply_tone_curve_batched(&mut batch, ctx, image, options, pixel_count)?;
    }

    // Stage 14: HSL adjustments (if scan profile has them)
    if let Some(ref profile) = options.scan_profile {
        if let Some(ref hsl) = profile.hsl_adjustments {
            if hsl.has_adjustments() {
                apply_hsl_adjustments_batched(&mut batch, ctx, image, hsl, pixel_count)?;
            }
        }
    }

    // Stage 15: Clamp to working range
    if !options.no_clip {
        clamp_working_range_batched(&mut batch, ctx, image, pixel_count)?;
    }

    // ========================================================================
    // FINAL SYNC: Submit all remaining commands
    // ========================================================================
    batch.finish();

    Ok(())
}

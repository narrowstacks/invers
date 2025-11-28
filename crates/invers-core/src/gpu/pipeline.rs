//! GPU pipeline orchestration for image processing.
//!
//! This module implements an efficient GPU pipeline that batches multiple
//! compute operations into single command buffer submissions to minimize
//! CPU-GPU synchronization overhead.

use wgpu;

use super::buffers::{
    create_uniform_buffer, ColorMatrixParams, FusedInvertParams, FusedPostprocessParams,
    GainParams, GpuHistogram, GpuImage, HslAdjustParams, InversionParams,
    ToneCurveParams as GpuToneCurveParams, UtilityParams, NUM_HISTOGRAM_BUCKETS,
};
use super::context::{GpuContext, GpuError};
use crate::decoders::DecodedImage;
use crate::models::{BaseEstimation, ConvertOptions, InversionMode, ShadowLiftMode};
use crate::pipeline::ProcessedImage;

/// Workgroup size for compute shaders
const WORKGROUP_SIZE: u32 = 256;

/// Maximum workgroups per dimension (GPU limit)
const MAX_WORKGROUPS_PER_DIM: u32 = 65535;

/// Process an image on the GPU.
///
/// Uses a cached GPU context that is initialized once and reused across all
/// image conversions. The first call may be slower due to shader compilation,
/// but subsequent calls benefit from cached pipelines.
pub fn process_image_gpu(
    decoded: &DecodedImage,
    options: &ConvertOptions,
) -> Result<ProcessedImage, GpuError> {
    // Get cached GPU context (initialized once, reused across all operations)
    let ctx = super::get_cached_context()?;

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

/// Execute the full processing pipeline on GPU using fused shaders.
///
/// The pipeline is optimized to minimize CPU-GPU sync points:
/// - Fast path (no auto-levels/auto-color): Single command buffer submission
/// - Slow path (with histogram): Split into 3 submissions with sync for histogram
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

    // Check if we need histogram (auto-levels or auto-color)
    let needs_histogram = options.enable_auto_levels
        || (options.enable_auto_color && !options.enable_auto_wb);

    if needs_histogram {
        // SLOW PATH: Need to sync for histogram analysis
        execute_pipeline_with_histogram(ctx, image, histogram, &base, options, pixel_count)
    } else {
        // FAST PATH: Single submission, no sync until final wait
        execute_pipeline_fast(ctx, image, &base, options, pixel_count)
    }
}

/// Fast path: Single command buffer with all operations.
fn execute_pipeline_fast(
    ctx: &GpuContext,
    image: &mut GpuImage,
    base: &BaseEstimation,
    options: &ConvertOptions,
    pixel_count: u32,
) -> Result<(), GpuError> {
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("pipeline_fast_encoder"),
        });

    // Inversion pass
    encode_fused_invert(ctx, &mut encoder, image, base, options, pixel_count)?;

    // Post-processing pass (no auto gains)
    encode_fused_postprocess(
        ctx,
        &mut encoder,
        image,
        options,
        None, // no auto-levels
        None, // no auto-color
        pixel_count,
    )?;

    // HSL adjustments (if needed)
    if let Some(ref profile) = options.scan_profile {
        if let Some(ref hsl) = profile.hsl_adjustments {
            if hsl.has_adjustments() {
                encode_hsl_adjustments(ctx, &mut encoder, image, hsl, pixel_count)?;
            }
        }
    }

    // Single submission
    ctx.queue.submit(std::iter::once(encoder.finish()));
    ctx.device.poll(wgpu::Maintain::Wait);

    Ok(())
}

/// Slow path: Multiple submissions with histogram sync.
fn execute_pipeline_with_histogram(
    ctx: &GpuContext,
    image: &mut GpuImage,
    histogram: &GpuHistogram,
    base: &BaseEstimation,
    options: &ConvertOptions,
    pixel_count: u32,
) -> Result<(), GpuError> {
    // BATCH 1: Fused inversion
    {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fused_invert_encoder"),
            });

        encode_fused_invert(ctx, &mut encoder, image, base, options, pixel_count)?;
        ctx.queue.submit(std::iter::once(encoder.finish()));
    }

    // Wait for inversion to complete before histogram
    ctx.device.poll(wgpu::Maintain::Wait);

    // Build histogram
    execute_histogram(ctx, image, histogram)?;

    // Download histogram and compute gains
    let [hist_r, hist_g, hist_b] = histogram.download()?;

    let auto_levels_gains = if options.enable_auto_levels {
        Some(compute_auto_levels_gains(
            &hist_r,
            &hist_g,
            &hist_b,
            options.auto_levels_clip_percent,
        ))
    } else {
        None
    };

    let auto_color_gains = if options.enable_auto_color && !options.enable_auto_wb {
        let gains = compute_auto_color_gains(&hist_r, &hist_g, &hist_b, options);
        let strength = options.auto_color_strength;
        Some([
            1.0 + strength * (gains[0] - 1.0),
            1.0 + strength * (gains[1] - 1.0),
            1.0 + strength * (gains[2] - 1.0),
        ])
    } else {
        None
    };

    // BATCH 2: Post-processing with computed gains
    {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fused_postprocess_encoder"),
            });

        encode_fused_postprocess(
            ctx,
            &mut encoder,
            image,
            options,
            auto_levels_gains,
            auto_color_gains,
            pixel_count,
        )?;

        // HSL adjustments (if needed)
        if let Some(ref profile) = options.scan_profile {
            if let Some(ref hsl) = profile.hsl_adjustments {
                if hsl.has_adjustments() {
                    encode_hsl_adjustments(ctx, &mut encoder, image, hsl, pixel_count)?;
                }
            }
        }

        ctx.queue.submit(std::iter::once(encoder.finish()));
        ctx.device.poll(wgpu::Maintain::Wait);
    }

    Ok(())
}

/// Execute histogram clear + accumulate as a batch, then wait.
fn execute_histogram(
    ctx: &GpuContext,
    image: &GpuImage,
    histogram: &GpuHistogram,
) -> Result<(), GpuError> {
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("histogram_encoder"),
        });

    encode_histogram_clear(ctx, &mut encoder, histogram)?;
    encode_histogram_accumulate(ctx, &mut encoder, image, histogram)?;

    ctx.queue.submit(std::iter::once(encoder.finish()));
    ctx.device.poll(wgpu::Maintain::Wait);
    Ok(())
}

// ============================================================================
// Individual encode operations (kept for backwards compatibility and debugging)
// ============================================================================

/// Default headroom for B&W mode: 5% of base value preserved as shadow detail
const BW_DEFAULT_HEADROOM: f32 = 0.05;

#[allow(dead_code)]
/// Encode inversion operation to command buffer.
fn encode_inversion(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    image: &GpuImage,
    base: &BaseEstimation,
    mode: &InversionMode,
    pixel_count: u32,
) -> Result<(), GpuError> {
    let (green_floor, blue_floor) = if let Some(ref mask) = base.mask_profile {
        let (_red, green, blue) = mask.calculate_shadow_floors();
        (green, blue)
    } else {
        (0.0, 0.0)
    };

    let params = InversionParams {
        base_r: base.medians[0],
        base_g: base.medians[1],
        base_b: base.medians[2],
        green_floor,
        blue_floor,
        bw_headroom: BW_DEFAULT_HEADROOM,
        pixel_count,
        _padding: 0,
    };

    let uniform_buffer = create_uniform_buffer(&ctx.device, &params, "inversion_params");

    let pipeline = match mode {
        InversionMode::Linear => &ctx.pipelines.inversion_linear,
        InversionMode::Logarithmic => &ctx.pipelines.inversion_log,
        InversionMode::DivideBlend => &ctx.pipelines.inversion_divide,
        InversionMode::MaskAware => &ctx.pipelines.inversion_mask_aware,
        InversionMode::BlackAndWhite => &ctx.pipelines.inversion_bw,
    };

    encode_compute_pass(ctx, encoder, pipeline, &image.buffer, &uniform_buffer, pixel_count)
}

#[allow(dead_code)]
/// Encode shadow lift operation to command buffer.
fn encode_shadow_lift(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    image: &GpuImage,
    options: &ConvertOptions,
    pixel_count: u32,
) -> Result<(), GpuError> {
    let lift = match options.shadow_lift_mode {
        ShadowLiftMode::Fixed => options.shadow_lift_value,
        ShadowLiftMode::Percentile => options.shadow_lift_value,
        ShadowLiftMode::None => return Ok(()),
    };

    let params = UtilityParams {
        param1: lift,
        param2: 0.0,
        param3: 0.0,
        pixel_count,
    };

    let uniform_buffer = create_uniform_buffer(&ctx.device, &params, "shadow_lift_params");
    encode_compute_pass(ctx, encoder, &ctx.pipelines.shadow_lift, &image.buffer, &uniform_buffer, pixel_count)
}

#[allow(dead_code)]
/// Encode highlight compression operation to command buffer.
fn encode_highlight_compression(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    image: &GpuImage,
    threshold: f32,
    compression: f32,
    pixel_count: u32,
) -> Result<(), GpuError> {
    let params = UtilityParams {
        param1: threshold,
        param2: compression,
        param3: 0.0,
        pixel_count,
    };

    let uniform_buffer = create_uniform_buffer(&ctx.device, &params, "highlight_compress_params");
    encode_compute_pass(ctx, encoder, &ctx.pipelines.highlight_compress, &image.buffer, &uniform_buffer, pixel_count)
}

#[allow(dead_code)]
/// Encode gains application to command buffer.
fn encode_gains(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    image: &GpuImage,
    gains: [f32; 3],
    offsets: [f32; 3],
    pixel_count: u32,
) -> Result<(), GpuError> {
    let params = GainParams {
        gain_r: gains[0],
        gain_g: gains[1],
        gain_b: gains[2],
        offset_r: offsets[0],
        offset_g: offsets[1],
        offset_b: offsets[2],
        pixel_count,
        _padding: 0,
    };

    let uniform_buffer = create_uniform_buffer(&ctx.device, &params, "gain_params");
    encode_compute_pass(ctx, encoder, &ctx.pipelines.apply_gains, &image.buffer, &uniform_buffer, pixel_count)
}

#[allow(dead_code)]
/// Encode base offsets application to command buffer.
fn encode_base_offsets(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    image: &GpuImage,
    offsets: [f32; 3],
    pixel_count: u32,
) -> Result<(), GpuError> {
    let params = GainParams {
        gain_r: 1.0,
        gain_g: 1.0,
        gain_b: 1.0,
        offset_r: -offsets[0],
        offset_g: -offsets[1],
        offset_b: -offsets[2],
        pixel_count,
        _padding: 0,
    };

    let uniform_buffer = create_uniform_buffer(&ctx.device, &params, "base_offset_params");
    encode_compute_pass(ctx, encoder, &ctx.pipelines.apply_gains, &image.buffer, &uniform_buffer, pixel_count)
}

#[allow(dead_code)]
/// Encode exposure multiplication to command buffer.
fn encode_exposure(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    image: &GpuImage,
    multiplier: f32,
    max_value: f32,
    pixel_count: u32,
) -> Result<(), GpuError> {
    let params = UtilityParams {
        param1: multiplier,
        param2: max_value,
        param3: 0.0,
        pixel_count,
    };

    let uniform_buffer = create_uniform_buffer(&ctx.device, &params, "exposure_params");
    encode_compute_pass(ctx, encoder, &ctx.pipelines.exposure_multiply, &image.buffer, &uniform_buffer, pixel_count)
}

#[allow(dead_code)]
/// Encode color matrix application to command buffer.
fn encode_color_matrix(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    image: &GpuImage,
    matrix: &[[f32; 3]; 3],
    pixel_count: u32,
) -> Result<(), GpuError> {
    let params = ColorMatrixParams {
        m00: matrix[0][0],
        m01: matrix[0][1],
        m02: matrix[0][2],
        _pad0: 0.0,
        m10: matrix[1][0],
        m11: matrix[1][1],
        m12: matrix[1][2],
        _pad1: 0.0,
        m20: matrix[2][0],
        m21: matrix[2][1],
        m22: matrix[2][2],
        _pad2: 0.0,
        pixel_count,
        _padding: [0, 0, 0],
    };

    let uniform_buffer = create_uniform_buffer(&ctx.device, &params, "color_matrix_params");
    encode_compute_pass(ctx, encoder, &ctx.pipelines.color_matrix, &image.buffer, &uniform_buffer, pixel_count)
}

#[allow(dead_code)]
/// Encode tone curve application to command buffer.
fn encode_tone_curve(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    image: &GpuImage,
    options: &ConvertOptions,
    pixel_count: u32,
) -> Result<(), GpuError> {
    let curve_params = options.film_preset.as_ref().map(|p| &p.tone_curve);

    let (pipeline, params) = match curve_params {
        Some(curve) => {
            match curve.curve_type.as_str() {
                "linear" => return Ok(()),
                "scurve" | "neutral" | "s-curve" => {
                    let p = GpuToneCurveParams {
                        strength: curve.strength,
                        toe_strength: 0.0,
                        toe_length: 0.0,
                        shoulder_strength: 0.0,
                        shoulder_start: 0.0,
                        pixel_count,
                        _padding: [0, 0],
                    };
                    (&ctx.pipelines.tone_curve_scurve, p)
                }
                "asymmetric" => {
                    let p = GpuToneCurveParams {
                        strength: curve.strength,
                        toe_strength: curve.toe_strength,
                        toe_length: curve.toe_length,
                        shoulder_strength: curve.shoulder_strength,
                        shoulder_start: curve.shoulder_start,
                        pixel_count,
                        _padding: [0, 0],
                    };
                    (&ctx.pipelines.tone_curve_asymmetric, p)
                }
                _ => {
                    let p = GpuToneCurveParams {
                        strength: curve.strength,
                        toe_strength: 0.0,
                        toe_length: 0.0,
                        shoulder_strength: 0.0,
                        shoulder_start: 0.0,
                        pixel_count,
                        _padding: [0, 0],
                    };
                    (&ctx.pipelines.tone_curve_scurve, p)
                }
            }
        }
        None => {
            let p = GpuToneCurveParams {
                strength: 0.3,
                toe_strength: 0.0,
                toe_length: 0.0,
                shoulder_strength: 0.0,
                shoulder_start: 0.0,
                pixel_count,
                _padding: [0, 0],
            };
            (&ctx.pipelines.tone_curve_scurve, p)
        }
    };

    let uniform_buffer = create_uniform_buffer(&ctx.device, &params, "tone_curve_params");
    encode_compute_pass(ctx, encoder, pipeline, &image.buffer, &uniform_buffer, pixel_count)
}

/// Encode HSL adjustments to command buffer (still used by fused pipeline).
fn encode_hsl_adjustments(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    image: &GpuImage,
    hsl: &crate::models::HslAdjustments,
    pixel_count: u32,
) -> Result<(), GpuError> {
    let params = HslAdjustParams {
        hue_adj_0: [hsl.hue[0], hsl.hue[1], hsl.hue[2], hsl.hue[3]],
        hue_adj_1: [hsl.hue[4], hsl.hue[5], hsl.hue[6], hsl.hue[7]],
        sat_adj_0: [hsl.saturation[0], hsl.saturation[1], hsl.saturation[2], hsl.saturation[3]],
        sat_adj_1: [hsl.saturation[4], hsl.saturation[5], hsl.saturation[6], hsl.saturation[7]],
        lum_adj_0: [hsl.luminance[0], hsl.luminance[1], hsl.luminance[2], hsl.luminance[3]],
        lum_adj_1: [hsl.luminance[4], hsl.luminance[5], hsl.luminance[6], hsl.luminance[7]],
        pixel_count,
        _padding: [0, 0, 0],
    };

    let uniform_buffer = create_uniform_buffer(&ctx.device, &params, "hsl_adjust_params");
    encode_compute_pass(ctx, encoder, &ctx.pipelines.hsl_adjust, &image.buffer, &uniform_buffer, pixel_count)
}

#[allow(dead_code)]
/// Encode clamp to working range to command buffer.
fn encode_clamp_range(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    image: &GpuImage,
    pixel_count: u32,
) -> Result<(), GpuError> {
    let params = UtilityParams {
        param1: 0.0,
        param2: 0.0,
        param3: 0.0,
        pixel_count,
    };

    let uniform_buffer = create_uniform_buffer(&ctx.device, &params, "clamp_params");
    encode_compute_pass(ctx, encoder, &ctx.pipelines.clamp_range, &image.buffer, &uniform_buffer, pixel_count)
}

// ============================================================================
// Fused shader operations (optimized multi-operation passes)
// ============================================================================

/// Encode fused inversion operation (invert + shadow lift + highlight compress).
fn encode_fused_invert(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    image: &GpuImage,
    base: &BaseEstimation,
    options: &ConvertOptions,
    pixel_count: u32,
) -> Result<(), GpuError> {
    // Calculate shadow floors from mask profile
    let (green_floor, blue_floor) = if let Some(ref mask) = base.mask_profile {
        let (_red, green, blue) = mask.calculate_shadow_floors();
        (green, blue)
    } else {
        (0.0, 0.0)
    };

    // Determine inversion mode flag
    let mode_flag = match options.inversion_mode {
        InversionMode::Linear => FusedInvertParams::MODE_LINEAR,
        InversionMode::Logarithmic => FusedInvertParams::MODE_LOG,
        InversionMode::DivideBlend => FusedInvertParams::MODE_DIVIDE,
        InversionMode::MaskAware => FusedInvertParams::MODE_MASK_AWARE,
        InversionMode::BlackAndWhite => FusedInvertParams::MODE_BW,
    };

    // Build flags
    let mut flags = mode_flag;
    if options.shadow_lift_mode != ShadowLiftMode::None {
        flags |= FusedInvertParams::FLAG_SHADOW_LIFT;
    }
    if options.highlight_compression < 1.0 {
        flags |= FusedInvertParams::FLAG_HIGHLIGHT_COMPRESS;
    }

    // Get shadow lift value
    let shadow_lift = match options.shadow_lift_mode {
        ShadowLiftMode::Fixed | ShadowLiftMode::Percentile => options.shadow_lift_value,
        ShadowLiftMode::None => 0.0,
    };

    let params = FusedInvertParams {
        base_r: base.medians[0],
        base_g: base.medians[1],
        base_b: base.medians[2],
        green_floor,
        blue_floor,
        bw_headroom: BW_DEFAULT_HEADROOM,
        shadow_lift,
        highlight_threshold: 0.9,
        highlight_compression: options.highlight_compression,
        flags,
        pixel_count,
        _padding: 0,
    };

    let uniform_buffer = create_uniform_buffer(&ctx.device, &params, "fused_invert_params");
    encode_compute_pass(
        ctx,
        encoder,
        &ctx.pipelines.fused_invert,
        &image.buffer,
        &uniform_buffer,
        pixel_count,
    )
}

/// Encode fused post-processing operation (gains + exposure + matrix + curve + clamp).
fn encode_fused_postprocess(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    image: &GpuImage,
    options: &ConvertOptions,
    auto_levels_gains: Option<([f32; 3], [f32; 3])>,
    auto_color_gains: Option<[f32; 3]>,
    pixel_count: u32,
) -> Result<(), GpuError> {
    let mut flags = 0u32;

    // Determine gains (auto-levels + auto-color + preset base offsets)
    let (gain_r, gain_g, gain_b, offset_r, offset_g, offset_b) =
        if let Some((gains, offsets)) = auto_levels_gains {
            flags |= FusedPostprocessParams::FLAG_GAINS;

            // Apply auto-color on top if present
            if let Some(color_gains) = auto_color_gains {
                (
                    gains[0] * color_gains[0],
                    gains[1] * color_gains[1],
                    gains[2] * color_gains[2],
                    offsets[0],
                    offsets[1],
                    offsets[2],
                )
            } else {
                (gains[0], gains[1], gains[2], offsets[0], offsets[1], offsets[2])
            }
        } else if let Some(color_gains) = auto_color_gains {
            flags |= FusedPostprocessParams::FLAG_GAINS;
            (color_gains[0], color_gains[1], color_gains[2], 0.0, 0.0, 0.0)
        } else {
            (1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
        };

    // Exposure
    let exposure_multiplier = if options.exposure_compensation != 1.0 {
        flags |= FusedPostprocessParams::FLAG_EXPOSURE;
        options.exposure_compensation
    } else {
        1.0
    };

    // Color matrix
    let (m00, m01, m02, m10, m11, m12, m20, m21, m22) = if !options.skip_color_matrix {
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
                flags |= FusedPostprocessParams::FLAG_COLOR_MATRIX;
                (
                    matrix[0][0], matrix[0][1], matrix[0][2],
                    matrix[1][0], matrix[1][1], matrix[1][2],
                    matrix[2][0], matrix[2][1], matrix[2][2],
                )
            } else {
                (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
            }
        } else {
            (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        }
    } else {
        (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    };

    // Tone curve
    let (tone_strength, toe_strength, toe_length, shoulder_strength, shoulder_start) =
        if !options.skip_tone_curve {
            if let Some(ref preset) = options.film_preset {
                let curve = &preset.tone_curve;
                match curve.curve_type.as_str() {
                    "linear" => (0.0, 0.0, 0.0, 0.0, 0.0),
                    "asymmetric" => {
                        flags |= FusedPostprocessParams::FLAG_TONE_ASYMMETRIC;
                        (
                            curve.strength,
                            curve.toe_strength,
                            curve.toe_length,
                            curve.shoulder_strength,
                            curve.shoulder_start,
                        )
                    }
                    _ => {
                        // Default to scurve
                        flags |= FusedPostprocessParams::FLAG_TONE_SCURVE;
                        (curve.strength, 0.0, 0.0, 0.0, 0.0)
                    }
                }
            } else {
                // Default tone curve
                flags |= FusedPostprocessParams::FLAG_TONE_SCURVE;
                (0.3, 0.0, 0.0, 0.0, 0.0)
            }
        } else {
            (0.0, 0.0, 0.0, 0.0, 0.0)
        };

    // Clamp
    if !options.no_clip {
        flags |= FusedPostprocessParams::FLAG_CLAMP;
    }

    let params = FusedPostprocessParams {
        gain_r,
        gain_g,
        gain_b,
        offset_r,
        offset_g,
        offset_b,
        m00, m01, m02,
        m10, m11, m12,
        m20, m21, m22,
        tone_strength,
        toe_strength,
        toe_length,
        shoulder_strength,
        shoulder_start,
        exposure_multiplier,
        flags,
        pixel_count,
    };

    let uniform_buffer = create_uniform_buffer(&ctx.device, &params, "fused_postprocess_params");
    encode_compute_pass(
        ctx,
        encoder,
        &ctx.pipelines.fused_postprocess,
        &image.buffer,
        &uniform_buffer,
        pixel_count,
    )
}

// ============================================================================
// Histogram operations (encode to command buffer)
// ============================================================================

/// Encode histogram clear to command buffer.
fn encode_histogram_clear(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    histogram: &GpuHistogram,
) -> Result<(), GpuError> {
    // Create a dummy buffer for binding 0 (pixels) since clear doesn't need it
    let dummy_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dummy_for_clear"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("histogram_clear_bind_group"),
        layout: &ctx.pipelines.histogram_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: dummy_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: histogram.buffer_r.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: histogram.buffer_g.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: histogram.buffer_b.as_entire_binding(),
            },
        ],
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("histogram_clear_pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&ctx.pipelines.histogram_clear);
        pass.set_bind_group(0, &bind_group, &[]);

        let workgroups = (NUM_HISTOGRAM_BUCKETS as u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    Ok(())
}

/// Encode histogram accumulation to command buffer.
fn encode_histogram_accumulate(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    image: &GpuImage,
    histogram: &GpuHistogram,
) -> Result<(), GpuError> {
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("histogram_accumulate_bind_group"),
        layout: &ctx.pipelines.histogram_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: image.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: histogram.buffer_r.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: histogram.buffer_g.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: histogram.buffer_b.as_entire_binding(),
            },
        ],
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("histogram_accumulate_pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&ctx.pipelines.histogram_accumulate);
        pass.set_bind_group(0, &bind_group, &[]);

        let pixel_count = image.pixel_count();
        let total_workgroups = (pixel_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        let (workgroups_x, workgroups_y) = compute_2d_dispatch(total_workgroups);
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    Ok(())
}

// ============================================================================
// Helper functions
// ============================================================================

/// Compute 2D dispatch dimensions for large workgroup counts.
fn compute_2d_dispatch(total_workgroups: u32) -> (u32, u32) {
    if total_workgroups <= MAX_WORKGROUPS_PER_DIM {
        (total_workgroups, 1)
    } else {
        let side = ((total_workgroups as f64).sqrt().ceil() as u32).min(MAX_WORKGROUPS_PER_DIM);
        let other = (total_workgroups + side - 1) / side;
        (side, other.min(MAX_WORKGROUPS_PER_DIM))
    }
}

/// Encode a compute pass to the command buffer using cached bind group layout.
fn encode_compute_pass(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    storage_buffer: &wgpu::Buffer,
    uniform_buffer: &wgpu::Buffer,
    pixel_count: u32,
) -> Result<(), GpuError> {
    // Use cached bind group layout from context
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compute_bind_group"),
        layout: &ctx.pipelines.storage_uniform_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: storage_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    });

    let total_workgroups = (pixel_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    let (workgroups_x, workgroups_y) = compute_2d_dispatch(total_workgroups);

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compute_pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    Ok(())
}

/// CPU-side base estimation (statistical analysis)
fn estimate_base_cpu(decoded: &DecodedImage, _options: &ConvertOptions) -> BaseEstimation {
    // Delegate to the existing CPU implementation
    crate::pipeline::estimate_base(decoded, None, None, None)
        .unwrap_or_else(|_| BaseEstimation {
            roi: None,
            medians: [0.5, 0.5, 0.5],
            noise_stats: None,
            auto_estimated: true,
            mask_profile: None,
        })
}

/// Compute auto-levels gains from histogram data
fn compute_auto_levels_gains(
    hist_r: &[u32],
    hist_g: &[u32],
    hist_b: &[u32],
    clip_percent: f32,
) -> ([f32; 3], [f32; 3]) {
    fn find_percentile_bounds(hist: &[u32], clip: f32) -> (f32, f32) {
        let total: u64 = hist.iter().map(|&x| x as u64).sum();
        let clip_count = (total as f32 * clip / 100.0) as u64;

        let mut low_sum: u64 = 0;
        let mut low_idx = 0;
        for (i, &count) in hist.iter().enumerate() {
            low_sum += count as u64;
            if low_sum >= clip_count {
                low_idx = i;
                break;
            }
        }

        let mut high_sum: u64 = 0;
        let mut high_idx = hist.len() - 1;
        for (i, &count) in hist.iter().enumerate().rev() {
            high_sum += count as u64;
            if high_sum >= clip_count {
                high_idx = i;
                break;
            }
        }

        // Use (len - 1) to match CPU histogram bucket normalization
        let max_bucket = (hist.len() - 1) as f32;
        (low_idx as f32 / max_bucket, high_idx as f32 / max_bucket)
    }

    let (min_r, max_r) = find_percentile_bounds(hist_r, clip_percent);
    let (min_g, max_g) = find_percentile_bounds(hist_g, clip_percent);
    let (min_b, max_b) = find_percentile_bounds(hist_b, clip_percent);

    let gains = [
        1.0 / (max_r - min_r).max(0.001),
        1.0 / (max_g - min_g).max(0.001),
        1.0 / (max_b - min_b).max(0.001),
    ];

    let offsets = [min_r, min_g, min_b];

    (gains, offsets)
}

/// Compute auto-color gains from histogram
fn compute_auto_color_gains(
    hist_r: &[u32],
    hist_g: &[u32],
    hist_b: &[u32],
    options: &ConvertOptions,
) -> [f32; 3] {
    // Find average values in midtone range
    fn compute_weighted_average(hist: &[u32], low: f32, high: f32) -> f32 {
        let buckets = hist.len();
        let low_idx = (low * buckets as f32) as usize;
        let high_idx = (high * buckets as f32) as usize;

        let mut sum: f64 = 0.0;
        let mut count: u64 = 0;

        for i in low_idx..=high_idx.min(buckets - 1) {
            let value = i as f64 / buckets as f64;
            sum += value * hist[i] as f64;
            count += hist[i] as u64;
        }

        if count > 0 {
            (sum / count as f64) as f32
        } else {
            0.5
        }
    }

    let avg_r = compute_weighted_average(hist_r, 0.35, 0.65);
    let avg_g = compute_weighted_average(hist_g, 0.35, 0.65);
    let avg_b = compute_weighted_average(hist_b, 0.35, 0.65);

    let target = (avg_r + avg_g + avg_b) / 3.0;

    let min_gain = options.auto_color_min_gain;
    let max_gain = options.auto_color_max_gain;

    [
        (target / avg_r.max(0.001)).clamp(min_gain, max_gain),
        (target / avg_g.max(0.001)).clamp(min_gain, max_gain),
        (target / avg_b.max(0.001)).clamp(min_gain, max_gain),
    ]
}

// Note: Auto-WB and auto-exposure are disabled in GPU mode because they require
// downloading the entire image (~480MB for 40MP) for statistical analysis.
// These features work in CPU mode. Future optimization could implement histogram-based
// approximations on GPU.

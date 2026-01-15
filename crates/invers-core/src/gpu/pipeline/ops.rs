//! Non-batched GPU operations (kept for testing/debugging).

use super::dispatch::dispatch_compute;
use crate::gpu::buffers::{
    create_uniform_buffer, CbInversionParams, CbLayerParams, ColorMatrixParams, GainParams,
    GpuImage, HslAdjustParams, InversionParams, ToneCurveParams, UtilityParams,
};
use crate::gpu::context::{GpuContext, GpuError};
use crate::models::{BaseEstimation, ConvertOptions, InversionMode, ShadowLiftMode};

/// Default headroom for B&W mode: 5% of base value preserved as shadow detail
pub const BW_DEFAULT_HEADROOM: f32 = 0.05;

#[allow(dead_code)]
pub fn apply_inversion(
    ctx: &GpuContext,
    image: &GpuImage,
    base: &BaseEstimation,
    mode: &InversionMode,
    pixel_count: u32,
) -> Result<(), GpuError> {
    // Calculate shadow floors for MaskAware mode
    let (green_floor, blue_floor) = if let Some(ref mask) = base.mask_profile {
        let (_red, green, blue) = mask.calculate_shadow_floors();
        (green, blue)
    } else {
        (0.0, 0.0)
    };

    // Pre-compute log10 values for log inversion optimization
    let log10_recip = 1.0 / 10.0_f32.ln();
    let params = InversionParams {
        base_r: base.medians[0],
        base_g: base.medians[1],
        base_b: base.medians[2],
        green_floor,
        blue_floor,
        bw_headroom: BW_DEFAULT_HEADROOM,
        pixel_count,
        _padding: 0,
        log_base_r: base.medians[0].max(0.0001).ln() * log10_recip,
        log_base_g: base.medians[1].max(0.0001).ln() * log10_recip,
        log_base_b: base.medians[2].max(0.0001).ln() * log10_recip,
        _padding2: 0.0,
    };

    let uniform_buffer = create_uniform_buffer(&ctx.device, &params, "inversion_params");

    // Select pipeline based on mode
    let pipeline = match mode {
        InversionMode::Linear => &ctx.pipelines.inversion_linear,
        InversionMode::Logarithmic => &ctx.pipelines.inversion_log,
        InversionMode::DivideBlend => &ctx.pipelines.inversion_divide,
        InversionMode::MaskAware => &ctx.pipelines.inversion_mask_aware,
        InversionMode::BlackAndWhite => &ctx.pipelines.inversion_bw,
    };

    dispatch_compute(ctx, pipeline, &image.buffer, &uniform_buffer, pixel_count)
}

#[allow(dead_code)]
pub fn apply_cb_inversion(
    ctx: &GpuContext,
    image: &GpuImage,
    analysis: &crate::models::CbHistogramAnalysis,
    is_negative: bool,
    pixel_count: u32,
) -> Result<(), GpuError> {
    let params = CbInversionParams {
        white_r: analysis.red.white_point,
        black_r: analysis.red.black_point,
        white_g: analysis.green.white_point,
        black_g: analysis.green.black_point,
        white_b: analysis.blue.white_point,
        black_b: analysis.blue.black_point,
        is_negative: if is_negative { 1 } else { 0 },
        pixel_count,
    };

    let uniform_buffer = create_uniform_buffer(&ctx.device, &params, "cb_inversion_params");
    dispatch_compute(
        ctx,
        &ctx.pipelines.cb_inversion,
        &image.buffer,
        &uniform_buffer,
        pixel_count,
    )
}

#[allow(dead_code)]
pub fn apply_shadow_lift(
    ctx: &GpuContext,
    image: &GpuImage,
    options: &ConvertOptions,
    pixel_count: u32,
) -> Result<(), GpuError> {
    let lift = match options.shadow_lift_mode {
        ShadowLiftMode::Fixed => options.shadow_lift_value,
        ShadowLiftMode::Percentile => {
            // For percentile mode, we'd need to compute the percentile first
            // For now, use fixed mode on GPU
            options.shadow_lift_value
        }
        ShadowLiftMode::None => return Ok(()),
    };

    let params = UtilityParams {
        param1: lift,
        param2: 0.0,
        param3: 0.0,
        pixel_count,
    };

    let uniform_buffer = create_uniform_buffer(&ctx.device, &params, "shadow_lift_params");
    dispatch_compute(
        ctx,
        &ctx.pipelines.shadow_lift,
        &image.buffer,
        &uniform_buffer,
        pixel_count,
    )
}

#[allow(dead_code)]
pub fn apply_highlight_compression(
    ctx: &GpuContext,
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
    dispatch_compute(
        ctx,
        &ctx.pipelines.highlight_compress,
        &image.buffer,
        &uniform_buffer,
        pixel_count,
    )
}

#[allow(dead_code)]
pub fn apply_gains(
    ctx: &GpuContext,
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
    dispatch_compute(
        ctx,
        &ctx.pipelines.apply_gains,
        &image.buffer,
        &uniform_buffer,
        pixel_count,
    )
}

#[allow(dead_code)]
pub fn apply_base_offsets(
    ctx: &GpuContext,
    image: &GpuImage,
    offsets: [f32; 3],
    pixel_count: u32,
) -> Result<(), GpuError> {
    // Reuse apply_gains with gain=1.0 and negated offsets
    // apply_gains does (value - offset) * gain, so we negate to get value + offset
    let gain_params = GainParams {
        gain_r: 1.0,
        gain_g: 1.0,
        gain_b: 1.0,
        offset_r: -offsets[0], // Negate because apply_gains does (value - offset) * gain
        offset_g: -offsets[1],
        offset_b: -offsets[2],
        pixel_count,
        _padding: 0,
    };

    let uniform_buffer = create_uniform_buffer(&ctx.device, &gain_params, "base_offset_params");
    dispatch_compute(
        ctx,
        &ctx.pipelines.apply_gains,
        &image.buffer,
        &uniform_buffer,
        pixel_count,
    )
}

#[allow(dead_code)]
pub fn apply_exposure(
    ctx: &GpuContext,
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
    dispatch_compute(
        ctx,
        &ctx.pipelines.exposure_multiply,
        &image.buffer,
        &uniform_buffer,
        pixel_count,
    )
}

#[allow(dead_code)]
pub fn apply_color_matrix(
    ctx: &GpuContext,
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
    dispatch_compute(
        ctx,
        &ctx.pipelines.color_matrix,
        &image.buffer,
        &uniform_buffer,
        pixel_count,
    )
}

#[allow(dead_code)]
pub fn apply_tone_curve(
    ctx: &GpuContext,
    image: &GpuImage,
    options: &ConvertOptions,
    pixel_count: u32,
) -> Result<(), GpuError> {
    // Get tone curve from film preset if available
    let curve_params = options.film_preset.as_ref().map(|p| &p.tone_curve);

    let (pipeline, params) = match curve_params {
        Some(curve) => {
            // curve_type is a String: "linear", "neutral", "scurve", "asymmetric"
            match curve.curve_type.as_str() {
                "linear" => return Ok(()), // No-op
                "scurve" | "neutral" | "s-curve" => {
                    let p = ToneCurveParams {
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
                    let p = ToneCurveParams {
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
                    // Default to S-curve for unknown types
                    let p = ToneCurveParams {
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
            // Default neutral S-curve
            let p = ToneCurveParams {
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
    dispatch_compute(ctx, pipeline, &image.buffer, &uniform_buffer, pixel_count)
}

#[allow(dead_code)]
pub fn apply_cb_layers(
    ctx: &GpuContext,
    image: &GpuImage,
    params: &CbLayerParams,
    pixel_count: u32,
) -> Result<(), GpuError> {
    let uniform_buffer = create_uniform_buffer(&ctx.device, params, "cb_layer_params");
    dispatch_compute(
        ctx,
        &ctx.pipelines.cb_layers,
        &image.buffer,
        &uniform_buffer,
        pixel_count,
    )
}

#[allow(dead_code)]
pub fn apply_hsl_adjustments(
    ctx: &GpuContext,
    image: &GpuImage,
    hsl: &crate::models::HslAdjustments,
    pixel_count: u32,
) -> Result<(), GpuError> {
    // HslAdjustments has arrays: hue[8], saturation[8], luminance[8]
    // Order: [R, O, Y, G, A, B, P, M]
    // Split into vec4 pairs for WGSL alignment
    let params = HslAdjustParams {
        hue_adj_0: [hsl.hue[0], hsl.hue[1], hsl.hue[2], hsl.hue[3]],
        hue_adj_1: [hsl.hue[4], hsl.hue[5], hsl.hue[6], hsl.hue[7]],
        sat_adj_0: [
            hsl.saturation[0],
            hsl.saturation[1],
            hsl.saturation[2],
            hsl.saturation[3],
        ],
        sat_adj_1: [
            hsl.saturation[4],
            hsl.saturation[5],
            hsl.saturation[6],
            hsl.saturation[7],
        ],
        lum_adj_0: [
            hsl.luminance[0],
            hsl.luminance[1],
            hsl.luminance[2],
            hsl.luminance[3],
        ],
        lum_adj_1: [
            hsl.luminance[4],
            hsl.luminance[5],
            hsl.luminance[6],
            hsl.luminance[7],
        ],
        pixel_count,
        _padding: [0, 0, 0],
    };

    let uniform_buffer = create_uniform_buffer(&ctx.device, &params, "hsl_adjust_params");
    dispatch_compute(
        ctx,
        &ctx.pipelines.hsl_adjust,
        &image.buffer,
        &uniform_buffer,
        pixel_count,
    )
}

#[allow(dead_code)]
pub fn clamp_working_range(
    ctx: &GpuContext,
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
    dispatch_compute(
        ctx,
        &ctx.pipelines.clamp_range,
        &image.buffer,
        &uniform_buffer,
        pixel_count,
    )
}

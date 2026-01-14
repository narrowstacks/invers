//! GPU pipeline orchestration for image processing.

use wgpu;

use super::buffers::{
    create_uniform_buffer, CbInversionParams, CbLayerParams, ColorMatrixParams, GainParams,
    GpuHistogram, GpuImage, HslAdjustParams, InversionParams, SubsampleParams,
    ToneCurveParams as GpuToneCurveParams, UtilityParams, NUM_HISTOGRAM_BUCKETS,
};
use super::context::{GpuContext, GpuError};
use crate::cb_pipeline::{
    analyze_histogram, analyze_wb_points, calculate_wb_gamma, calculate_wb_offsets,
    calculate_wb_preset_offsets,
};
use crate::decoders::DecodedImage;
use crate::models::{
    AutoWbMode, BaseEstimation, CbLayerOrder, CbWbMethod, CbWbPreset, ConvertOptions,
    InversionMode, ShadowLiftMode,
};
use crate::pipeline::ProcessedImage;

/// Workgroup size for compute shaders
const WORKGROUP_SIZE: u32 = 256;

// ============================================================================
// Command Batching Infrastructure
// ============================================================================

/// A batch of GPU commands that can be accumulated and submitted together.
///
/// Instead of submitting and waiting after each operation (which causes CPU-GPU
/// synchronization overhead), this struct accumulates compute passes into a single
/// command encoder. Commands are only submitted when:
/// 1. `flush()` is called (when data needs to be read back)
/// 2. `finish()` is called (at the end of the pipeline)
///
/// This can provide 2-4x speedup for pipelines with many stages.
pub struct CommandBatch<'a> {
    ctx: &'a GpuContext,
    encoder: Option<wgpu::CommandEncoder>,
    label: &'static str,
}

impl<'a> CommandBatch<'a> {
    /// Create a new command batch.
    pub fn new(ctx: &'a GpuContext, label: &'static str) -> Self {
        Self {
            ctx,
            encoder: Some(
                ctx.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) }),
            ),
            label,
        }
    }

    /// Get mutable access to the encoder for recording commands.
    /// Panics if the batch has already been finished.
    fn encoder_mut(&mut self) -> &mut wgpu::CommandEncoder {
        self.encoder
            .as_mut()
            .expect("CommandBatch already finished")
    }

    /// Submit all accumulated commands and wait for completion.
    /// Use this when you need to read data back from the GPU.
    /// After flushing, a new encoder is created for subsequent commands.
    pub fn flush(&mut self) {
        if let Some(encoder) = self.encoder.take() {
            self.ctx.queue.submit(std::iter::once(encoder.finish()));
            self.ctx.device.poll(wgpu::Maintain::Wait);

            // Create a new encoder for subsequent commands
            self.encoder = Some(self.ctx.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some(self.label),
                },
            ));
        }
    }

    /// Submit all accumulated commands and wait for completion.
    /// This consumes the batch - no more commands can be added.
    pub fn finish(mut self) {
        if let Some(encoder) = self.encoder.take() {
            self.ctx.queue.submit(std::iter::once(encoder.finish()));
            self.ctx.device.poll(wgpu::Maintain::Wait);
        }
    }

    /// Record a compute dispatch into the batch without submitting.
    pub fn dispatch(
        &mut self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroups_x: u32,
        workgroups_y: u32,
        label: &'static str,
    ) {
        let encoder = self.encoder_mut();
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }
}

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

// ============================================================================
// Individual operation implementations (non-batched - kept for testing/debugging)
// ============================================================================

/// Default headroom for B&W mode: 5% of base value preserved as shadow detail
const BW_DEFAULT_HEADROOM: f32 = 0.05;

#[allow(dead_code)]
fn apply_inversion(
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
fn apply_cb_inversion(
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
fn apply_shadow_lift(
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
fn apply_highlight_compression(
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

fn clear_histogram(ctx: &GpuContext, histogram: &GpuHistogram) -> Result<(), GpuError> {
    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("histogram_clear_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

    // Create a dummy buffer for binding 0 (pixels) since clear doesn't need it
    let dummy_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dummy_for_clear"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("histogram_clear_bind_group"),
        layout: &bind_group_layout,
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

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("histogram_clear_encoder"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("histogram_clear_pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&ctx.pipelines.histogram_clear);
        pass.set_bind_group(0, &bind_group, &[]);

        let workgroups = (NUM_HISTOGRAM_BUCKETS as u32).div_ceil(WORKGROUP_SIZE);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    ctx.submit_and_wait(encoder);
    Ok(())
}

fn accumulate_histogram(
    ctx: &GpuContext,
    image: &GpuImage,
    histogram: &GpuHistogram,
) -> Result<(), GpuError> {
    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("histogram_accumulate_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("histogram_accumulate_bind_group"),
        layout: &bind_group_layout,
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

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("histogram_accumulate_encoder"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("histogram_accumulate_pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&ctx.pipelines.histogram_accumulate);
        pass.set_bind_group(0, &bind_group, &[]);

        let pixel_count = image.pixel_count();
        let total_workgroups = pixel_count.div_ceil(WORKGROUP_SIZE);

        // Use 2D dispatch for large images
        let (workgroups_x, workgroups_y) = if total_workgroups <= MAX_WORKGROUPS_PER_DIM {
            (total_workgroups, 1)
        } else {
            let side = ((total_workgroups as f64).sqrt().ceil() as u32).min(MAX_WORKGROUPS_PER_DIM);
            let other = total_workgroups.div_ceil(side);
            (side, other.min(MAX_WORKGROUPS_PER_DIM))
        };
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    ctx.submit_and_wait(encoder);
    Ok(())
}

#[allow(dead_code)]
fn apply_gains(
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
fn apply_base_offsets(
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
fn apply_exposure(
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
fn apply_color_matrix(
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
fn apply_tone_curve(
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
                    // Default to S-curve for unknown types
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
            // Default neutral S-curve
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
    dispatch_compute(ctx, pipeline, &image.buffer, &uniform_buffer, pixel_count)
}

#[allow(dead_code)]
fn apply_cb_layers(
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
fn apply_hsl_adjustments(
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
fn clamp_working_range(
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

// ============================================================================
// Batched operation implementations (use CommandBatch instead of sync per-op)
// ============================================================================

fn apply_inversion_batched(
    batch: &mut CommandBatch,
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

    dispatch_compute_batched(
        batch,
        ctx,
        pipeline,
        &image.buffer,
        &uniform_buffer,
        pixel_count,
        "inversion_pass",
    )
}

fn apply_shadow_lift_batched(
    batch: &mut CommandBatch,
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
    dispatch_compute_batched(
        batch,
        ctx,
        &ctx.pipelines.shadow_lift,
        &image.buffer,
        &uniform_buffer,
        pixel_count,
        "shadow_lift_pass",
    )
}

fn apply_highlight_compression_batched(
    batch: &mut CommandBatch,
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
    dispatch_compute_batched(
        batch,
        ctx,
        &ctx.pipelines.highlight_compress,
        &image.buffer,
        &uniform_buffer,
        pixel_count,
        "highlight_compress_pass",
    )
}

fn apply_gains_batched(
    batch: &mut CommandBatch,
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
    dispatch_compute_batched(
        batch,
        ctx,
        &ctx.pipelines.apply_gains,
        &image.buffer,
        &uniform_buffer,
        pixel_count,
        "apply_gains_pass",
    )
}

fn apply_base_offsets_batched(
    batch: &mut CommandBatch,
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
    dispatch_compute_batched(
        batch,
        ctx,
        &ctx.pipelines.apply_gains,
        &image.buffer,
        &uniform_buffer,
        pixel_count,
        "base_offsets_pass",
    )
}

fn apply_exposure_batched(
    batch: &mut CommandBatch,
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
    dispatch_compute_batched(
        batch,
        ctx,
        &ctx.pipelines.exposure_multiply,
        &image.buffer,
        &uniform_buffer,
        pixel_count,
        "exposure_pass",
    )
}

fn apply_color_matrix_batched(
    batch: &mut CommandBatch,
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
    dispatch_compute_batched(
        batch,
        ctx,
        &ctx.pipelines.color_matrix,
        &image.buffer,
        &uniform_buffer,
        pixel_count,
        "color_matrix_pass",
    )
}

fn apply_tone_curve_batched(
    batch: &mut CommandBatch,
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
                    // Default to S-curve for unknown types
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
            // Default neutral S-curve
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
    dispatch_compute_batched(
        batch,
        ctx,
        pipeline,
        &image.buffer,
        &uniform_buffer,
        pixel_count,
        "tone_curve_pass",
    )
}

fn apply_hsl_adjustments_batched(
    batch: &mut CommandBatch,
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
    dispatch_compute_batched(
        batch,
        ctx,
        &ctx.pipelines.hsl_adjust,
        &image.buffer,
        &uniform_buffer,
        pixel_count,
        "hsl_adjust_pass",
    )
}

fn clamp_working_range_batched(
    batch: &mut CommandBatch,
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
    dispatch_compute_batched(
        batch,
        ctx,
        &ctx.pipelines.clamp_range,
        &image.buffer,
        &uniform_buffer,
        pixel_count,
        "clamp_range_pass",
    )
}

fn apply_cb_inversion_batched(
    batch: &mut CommandBatch,
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
    dispatch_compute_batched(
        batch,
        ctx,
        &ctx.pipelines.cb_inversion,
        &image.buffer,
        &uniform_buffer,
        pixel_count,
        "cb_inversion_pass",
    )
}

fn apply_cb_layers_batched(
    batch: &mut CommandBatch,
    ctx: &GpuContext,
    image: &GpuImage,
    params: &CbLayerParams,
    pixel_count: u32,
) -> Result<(), GpuError> {
    let uniform_buffer = create_uniform_buffer(&ctx.device, params, "cb_layer_params");
    dispatch_compute_batched(
        batch,
        ctx,
        &ctx.pipelines.cb_layers,
        &image.buffer,
        &uniform_buffer,
        pixel_count,
        "cb_layers_pass",
    )
}

// ============================================================================
// Helper functions
// ============================================================================

/// Maximum workgroups per dimension (GPU limit)
const MAX_WORKGROUPS_PER_DIM: u32 = 65535;

/// Maximum pixels per single dispatch (65535 workgroups * 256 threads)
#[allow(dead_code)]
const MAX_PIXELS_PER_DISPATCH: u32 = MAX_WORKGROUPS_PER_DIM * WORKGROUP_SIZE;

/// Generic compute dispatch for storage + uniform pattern
/// Handles large images by splitting into multiple dispatches when needed
#[allow(dead_code)]
fn dispatch_compute(
    ctx: &GpuContext,
    pipeline: &wgpu::ComputePipeline,
    storage_buffer: &wgpu::Buffer,
    uniform_buffer: &wgpu::Buffer,
    pixel_count: u32,
) -> Result<(), GpuError> {
    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compute_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compute_bind_group"),
        layout: &bind_group_layout,
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

    let total_workgroups = pixel_count.div_ceil(WORKGROUP_SIZE);

    // If within limits, do a single dispatch
    if total_workgroups <= MAX_WORKGROUPS_PER_DIM {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("compute_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute_pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(total_workgroups, 1, 1);
        }

        ctx.submit_and_wait(encoder);
    } else {
        // For large images, use 2D dispatch with both x and y dimensions
        // This allows up to 65535 * 65535 workgroups = ~4 billion workgroups
        // Calculate grid dimensions: try to make it roughly square for efficiency
        let side = ((total_workgroups as f64).sqrt().ceil() as u32).min(MAX_WORKGROUPS_PER_DIM);
        let workgroups_y = total_workgroups.div_ceil(side);

        if workgroups_y > MAX_WORKGROUPS_PER_DIM {
            return Err(GpuError::Other(format!(
                "Image too large: {} pixels requires {} workgroups, max supported is {}",
                pixel_count,
                total_workgroups,
                MAX_WORKGROUPS_PER_DIM as u64 * MAX_WORKGROUPS_PER_DIM as u64
            )));
        }

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("compute_encoder_2d"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute_pass_2d"),
                timestamp_writes: None,
            });

            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(side, workgroups_y, 1);
        }

        ctx.submit_and_wait(encoder);
    }

    Ok(())
}

/// Record a compute dispatch into a command batch without submitting.
/// Uses the cached bind group layout from GpuContext for efficiency.
fn dispatch_compute_batched(
    batch: &mut CommandBatch,
    ctx: &GpuContext,
    pipeline: &wgpu::ComputePipeline,
    storage_buffer: &wgpu::Buffer,
    uniform_buffer: &wgpu::Buffer,
    pixel_count: u32,
    label: &'static str,
) -> Result<(), GpuError> {
    // Use the cached bind group layout
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
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

    let total_workgroups = pixel_count.div_ceil(WORKGROUP_SIZE);

    // Calculate 2D dispatch dimensions for large images
    let (workgroups_x, workgroups_y) = if total_workgroups <= MAX_WORKGROUPS_PER_DIM {
        (total_workgroups, 1)
    } else {
        let side = ((total_workgroups as f64).sqrt().ceil() as u32).min(MAX_WORKGROUPS_PER_DIM);
        let y = total_workgroups.div_ceil(side);

        if y > MAX_WORKGROUPS_PER_DIM {
            return Err(GpuError::Other(format!(
                "Image too large: {} pixels requires {} workgroups, max supported is {}",
                pixel_count,
                total_workgroups,
                MAX_WORKGROUPS_PER_DIM as u64 * MAX_WORKGROUPS_PER_DIM as u64
            )));
        }
        (side, y)
    };

    batch.dispatch(pipeline, &bind_group, workgroups_x, workgroups_y, label);
    Ok(())
}

/// CPU-side base estimation (statistical analysis)
fn estimate_base_cpu(decoded: &DecodedImage, _options: &ConvertOptions) -> BaseEstimation {
    // Delegate to the existing CPU implementation
    crate::pipeline::estimate_base(decoded, None, None, None).unwrap_or(BaseEstimation {
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

/// Limit how much gains can diverge from each other to preserve scene character.
/// Mirrors the CPU implementation in auto_adjust.rs.
fn limit_channel_divergence(gains: [f32; 3], max_divergence: f32) -> [f32; 3] {
    let min_g = gains[0].min(gains[1]).min(gains[2]);
    let max_g = gains[0].max(gains[1]).max(gains[2]);
    let current_divergence = max_g - min_g;

    if current_divergence <= max_divergence {
        return gains; // Already within limits
    }

    // Scale gains toward their mean to reduce divergence while preserving relative proportions
    let mean_gain = (gains[0] + gains[1] + gains[2]) / 3.0;
    let scale = max_divergence / current_divergence;

    [
        mean_gain + (gains[0] - mean_gain) * scale,
        mean_gain + (gains[1] - mean_gain) * scale,
        mean_gain + (gains[2] - mean_gain) * scale,
    ]
}

/// Compute scene-adaptive auto-color gains using midtone-weighted means.
///
/// Uses a Gaussian weighting centered on midtones (luminance ~0.5) to compute
/// weighted channel averages. This ensures the color correction prioritizes
/// midtone neutrality, which is what the human eye is most sensitive to.
fn compute_auto_color_gains_fullimage(
    image: &GpuImage,
    _ctx: &GpuContext,
    options: &ConvertOptions,
) -> Result<[f32; 3], GpuError> {
    // Download image and compute midtone-weighted channel means
    let data = image.download()?;
    let mut r_sum = 0.0f64;
    let mut g_sum = 0.0f64;
    let mut b_sum = 0.0f64;
    let mut weight_sum = 0.0f64;

    for pixel in data.chunks_exact(3) {
        let r = pixel[0] as f64;
        let g = pixel[1] as f64;
        let b = pixel[2] as f64;

        // Compute luminance
        let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;

        // Gaussian weight centered at 0.5 with sigma ~0.25
        // This gives high weight to midtones, low weight to shadows/highlights
        let dist = lum - 0.5;
        let weight = (-dist * dist / (2.0 * 0.25 * 0.25)).exp();

        r_sum += r * weight;
        g_sum += g * weight;
        b_sum += b * weight;
        weight_sum += weight;
    }

    if weight_sum < 0.001 {
        return Ok([1.0, 1.0, 1.0]);
    }

    let avg_r = (r_sum / weight_sum) as f32;
    let avg_g = (g_sum / weight_sum) as f32;
    let avg_b = (b_sum / weight_sum) as f32;

    eprintln!(
        "[AUTO-COLOR] Midtone-weighted avgs: R={:.4} G={:.4} B={:.4}",
        avg_r, avg_g, avg_b
    );
    eprintln!(
        "[AUTO-COLOR] Ratios: R/G={:.4} B/G={:.4}",
        avg_r / avg_g,
        avg_b / avg_g
    );

    let strength = options.auto_color_strength;
    let min_gain = options.auto_color_min_gain;
    let max_gain = options.auto_color_max_gain;

    // Asymmetric color correction:
    // - Warmth (R > G) is often natural scene character - preserve it
    // - Coolness/blue cast (B > G) is often a scanning artifact - correct it
    //
    // Instead of targeting neutral R=G=B, we use G as reference and:
    // - Only reduce R if it's significantly above G (which is rare for natural scenes)
    // - Reduce B toward G when B > G (remove blue cast)
    // - Boost R/B toward G when they're below G (remove cyan/yellow casts)

    // Calculate what gain would make each channel equal to G
    let r_to_neutral = avg_g / avg_r.max(0.001);
    let b_to_neutral = avg_g / avg_b.max(0.001);

    // For R channel: if R > G (warm), boost warmth for film look
    // If R < G (cyan cast), use full strength to correct
    let r_is_warm = avg_r > avg_g;
    let r_gain = if r_is_warm {
        // Warm scene: boost warmth for pleasing film look, scaled by strength
        // Film inversions typically lose warmth - compensate with up to ~12% boost
        // This mimics the warmer tones of professional scanner output
        // At strength=0, no change (1.0); at strength=1.0, full boost (1.12)
        1.0 + strength * 0.12
    } else {
        // Cyan cast: boost R toward G
        1.0 + strength * (r_to_neutral - 1.0)
    };

    // For B channel: if B > G (blue cast), use full strength to correct
    // If B < G (yellow cast, rare), use reduced strength
    let b_is_blue = avg_b > avg_g;
    let b_strength = if b_is_blue { strength } else { strength * 0.3 };
    let b_gain = 1.0 + b_strength * (b_to_neutral - 1.0);

    eprintln!(
        "[AUTO-COLOR] r_is_warm={} (preserving) b_is_blue={} (correcting)",
        r_is_warm, b_is_blue
    );
    eprintln!("[AUTO-COLOR] b_strength={:.3}", b_strength);

    let gains = [
        r_gain.clamp(min_gain, max_gain),
        1.0, // G is reference, no adjustment
        b_gain.clamp(min_gain, max_gain),
    ];

    eprintln!(
        "[AUTO-COLOR] raw gains: R={:.4} G={:.4} B={:.4}",
        gains[0], gains[1], gains[2]
    );

    // Limit divergence to preserve scene character
    let final_gains = limit_channel_divergence(gains, options.auto_color_max_divergence);
    eprintln!(
        "[AUTO-COLOR] final gains (after divergence limit): R={:.4} G={:.4} B={:.4}",
        final_gains[0], final_gains[1], final_gains[2]
    );

    Ok(final_gains)
}

/// Compute scene-adaptive auto-color gains from histogram.
///
/// Uses the same algorithm as the CPU implementation:
/// - Scene-adaptive targeting (preserves scene warmth/coolness)
/// - Channel divergence limiting to prevent aggressive neutralization
#[allow(dead_code)]
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

        for (i, &bin_count) in hist
            .iter()
            .enumerate()
            .take(high_idx.min(buckets - 1) + 1)
            .skip(low_idx)
        {
            let value = i as f64 / buckets as f64;
            sum += value * bin_count as f64;
            count += bin_count as u64;
        }

        if count > 0 {
            (sum / count as f64) as f32
        } else {
            0.5
        }
    }

    // Use wider range (0.1-0.9) to capture more representative sample
    // The narrow midtone range (0.35-0.65) can miss highlights/shadows that
    // have different color distributions
    let avg_r = compute_weighted_average(hist_r, 0.1, 0.9);
    let avg_g = compute_weighted_average(hist_g, 0.1, 0.9);
    let avg_b = compute_weighted_average(hist_b, 0.1, 0.9);

    eprintln!(
        "[GPU AUTO-COLOR] Wide range avgs: R={:.4} G={:.4} B={:.4}",
        avg_r, avg_g, avg_b
    );

    // Scene-adaptive targeting: blend between scene-preserving and neutral
    let avg_luminance = (avg_r + avg_g + avg_b) / 3.0;
    let strength = options.auto_color_strength;

    eprintln!(
        "[GPU AUTO-COLOR] avg_luminance={:.4} strength={:.4}",
        avg_luminance, strength
    );

    // Target values that respect scene color temperature
    // At strength=1.0, we target neutral; at strength=0.0, we preserve scene completely
    let target_r = avg_r + strength * (avg_luminance - avg_r);
    let target_g = avg_g + strength * (avg_luminance - avg_g);
    let target_b = avg_b + strength * (avg_luminance - avg_b);

    eprintln!(
        "[GPU AUTO-COLOR] targets: R={:.4} G={:.4} B={:.4}",
        target_r, target_g, target_b
    );

    let min_gain = options.auto_color_min_gain;
    let max_gain = options.auto_color_max_gain;

    // Calculate gains with per-channel limits
    let gains = [
        (target_r / avg_r.max(0.001)).clamp(min_gain, max_gain),
        (target_g / avg_g.max(0.001)).clamp(min_gain, max_gain),
        (target_b / avg_b.max(0.001)).clamp(min_gain, max_gain),
    ];

    eprintln!(
        "[GPU AUTO-COLOR] raw gains: R={:.4} G={:.4} B={:.4}",
        gains[0], gains[1], gains[2]
    );
    eprintln!(
        "[GPU AUTO-COLOR] max_divergence={:.4}",
        options.auto_color_max_divergence
    );

    // Limit divergence to preserve scene character
    limit_channel_divergence(gains, options.auto_color_max_divergence)
}

/// Compute white balance gains (requires downloading image data)
fn compute_wb_gains_cpu(image: &GpuImage, _ctx: &GpuContext) -> Result<[f32; 3], GpuError> {
    // Download image data for WB computation
    let data = image.download()?;

    // Find highlight/gray pixels and compute gains
    let mut r_sum = 0.0f64;
    let mut g_sum = 0.0f64;
    let mut b_sum = 0.0f64;
    let mut count = 0u64;

    for pixel in data.chunks_exact(3) {
        let r = pixel[0];
        let g = pixel[1];
        let b = pixel[2];

        let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;

        // Use highlight pixels (bright areas likely to be neutral)
        if lum > 0.6 {
            r_sum += r as f64;
            g_sum += g as f64;
            b_sum += b as f64;
            count += 1;
        }
    }

    if count < 100 {
        return Ok([1.0, 1.0, 1.0]); // Not enough samples
    }

    let r_avg = (r_sum / count as f64) as f32;
    let g_avg = (g_sum / count as f64) as f32;
    let b_avg = (b_sum / count as f64) as f32;

    // Normalize to green channel
    Ok([g_avg / r_avg.max(0.001), 1.0, g_avg / b_avg.max(0.001)])
}

/// Compute exposure gain (requires downloading image data for median)
///
/// Uses a highlight-aware algorithm that:
/// 1. Computes the gain needed to reach target median
/// 2. Limits the gain to preserve highlights (98th percentile stays below 0.95)
fn compute_exposure_gain_cpu(
    image: &GpuImage,
    _ctx: &GpuContext,
    options: &ConvertOptions,
) -> Result<f32, GpuError> {
    let data = image.download()?;

    // Collect luminance values
    let mut luminances: Vec<f32> = data
        .chunks_exact(3)
        .map(|p| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2])
        .collect();

    if luminances.is_empty() {
        return Ok(1.0);
    }

    let n = luminances.len();

    // Find median (50th percentile)
    let mid = n / 2;
    let median = *luminances
        .select_nth_unstable_by(mid, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        })
        .1;

    // Find 98th percentile for highlight preservation
    let p98_idx = (n * 98) / 100;
    let p98 = *luminances
        .select_nth_unstable_by(p98_idx, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        })
        .1;

    // Compute gain to reach target median
    let target = options.auto_exposure_target_median;
    let median_gain = target / median.max(0.001);

    // Compute maximum gain that keeps 98th percentile below 0.95
    // This prevents highlight clipping
    const HIGHLIGHT_CEILING: f32 = 0.95;
    let highlight_limit_gain = if p98 > 0.001 {
        HIGHLIGHT_CEILING / p98
    } else {
        options.auto_exposure_max_gain
    };

    // Use the minimum of median-based gain and highlight-preserving gain
    let gain = median_gain.min(highlight_limit_gain);

    // Clamp gain to configured limits
    Ok(gain.clamp(
        options.auto_exposure_min_gain,
        options.auto_exposure_max_gain,
    ))
}

/// Download a subsampled version of the GPU image for efficient analysis.
///
/// Instead of downloading the full image (which can be 150MB+ for high-res scans),
/// this function runs a GPU shader to extract every Nth pixel, then downloads
/// only the subsampled result. For stride=8, this reduces transfer size by 64x.
///
/// The subsampled data is suitable for global analysis like white balance,
/// where exact pixel values aren't needed—statistical properties are preserved.
fn download_subsampled(
    ctx: &GpuContext,
    image: &GpuImage,
    stride: u32,
) -> Result<Vec<f32>, GpuError> {
    // Calculate output dimensions
    let output_width = image.width.div_ceil(stride);
    let output_height = image.height.div_ceil(stride);
    let output_pixel_count = output_width * output_height;
    let output_element_count = output_pixel_count * image.channels;

    // Create output buffer
    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("subsample_output"),
        size: (output_element_count as usize * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create uniform buffer with parameters
    let params = SubsampleParams {
        input_width: image.width,
        input_height: image.height,
        output_width,
        stride,
        output_pixel_count,
        _padding1: 0,
        _padding2: 0,
        _padding3: 0,
    };
    let uniform_buffer = create_uniform_buffer(&ctx.device, &params, "subsample_params");

    // Create bind group using the subsample layout
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("subsample_bind_group"),
        layout: &ctx.pipelines.subsample_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: image.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    });

    // Dispatch compute shader
    let total_workgroups = output_pixel_count.div_ceil(WORKGROUP_SIZE);

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("subsample_encoder"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("subsample_pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&ctx.pipelines.subsample);
        pass.set_bind_group(0, &bind_group, &[]);

        if total_workgroups <= MAX_WORKGROUPS_PER_DIM {
            pass.dispatch_workgroups(total_workgroups, 1, 1);
        } else {
            // Use 2D dispatch for very large outputs
            let side = ((total_workgroups as f64).sqrt().ceil() as u32).min(MAX_WORKGROUPS_PER_DIM);
            let workgroups_y = total_workgroups.div_ceil(side);
            pass.dispatch_workgroups(side, workgroups_y, 1);
        }
    }

    ctx.submit_and_wait(encoder);

    // Download the subsampled result
    let size = (output_element_count as usize * std::mem::size_of::<f32>()) as u64;
    let staging_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("subsample_staging"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut download_encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("subsample_download_encoder"),
        });

    download_encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, size);

    ctx.queue.submit(std::iter::once(download_encoder.finish()));

    // Map and read
    let buffer_slice = staging_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();

    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });

    ctx.device.poll(wgpu::Maintain::Wait);

    rx.recv()
        .map_err(|e| GpuError::BufferError(e.to_string()))?
        .map_err(|e| GpuError::BufferError(e.to_string()))?;

    let data = buffer_slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

    drop(data);
    staging_buffer.unmap();

    Ok(result)
}

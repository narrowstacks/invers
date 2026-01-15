//! GPU dispatch helpers and subsampling utilities.

use super::{CommandBatch, MAX_WORKGROUPS_PER_DIM, WORKGROUP_SIZE};
use crate::decoders::DecodedImage;
use crate::gpu::buffers::{create_uniform_buffer, GpuImage, SubsampleParams};
use crate::gpu::context::{GpuContext, GpuError};
use crate::models::{BaseEstimation, ConvertOptions};
use wgpu;

/// Generic compute dispatch for storage + uniform pattern
/// Handles large images by splitting into multiple dispatches when needed
#[allow(dead_code)]
pub fn dispatch_compute(
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
pub fn dispatch_compute_batched(
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
pub fn estimate_base_cpu(decoded: &DecodedImage, _options: &ConvertOptions) -> BaseEstimation {
    // Delegate to the existing CPU implementation
    crate::pipeline::estimate_base(decoded, None, None, None).unwrap_or(BaseEstimation {
        roi: None,
        medians: [0.5, 0.5, 0.5],
        noise_stats: None,
        auto_estimated: true,
        mask_profile: None,
    })
}

/// Download a subsampled version of the GPU image for efficient analysis.
///
/// Instead of downloading the full image (which can be 150MB+ for high-res scans),
/// this function runs a GPU shader to extract every Nth pixel, then downloads
/// only the subsampled result. For stride=8, this reduces transfer size by 64x.
///
/// The subsampled data is suitable for global analysis like white balance,
/// where exact pixel values aren't needed—statistical properties are preserved.
pub fn download_subsampled(
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

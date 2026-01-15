//! GPU histogram operations.

use super::{MAX_WORKGROUPS_PER_DIM, WORKGROUP_SIZE};
use crate::gpu::buffers::{GpuHistogram, GpuImage, NUM_HISTOGRAM_BUCKETS};
use crate::gpu::context::{GpuContext, GpuError};
use wgpu;

pub fn clear_histogram(ctx: &GpuContext, histogram: &GpuHistogram) -> Result<(), GpuError> {
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

pub fn accumulate_histogram(
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

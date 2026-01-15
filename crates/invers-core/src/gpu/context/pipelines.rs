//! Compute pipeline creation for GPU operations.

use super::GpuError;
use crate::gpu::shaders::Shaders;

/// Pre-compiled compute pipelines for all GPU operations.
#[allow(dead_code)]
pub struct GpuPipelines {
    // Inversion pipelines (one per mode)
    pub inversion_linear: wgpu::ComputePipeline,
    pub inversion_log: wgpu::ComputePipeline,
    pub inversion_divide: wgpu::ComputePipeline,
    pub inversion_mask_aware: wgpu::ComputePipeline,
    pub inversion_bw: wgpu::ComputePipeline,
    pub cb_inversion: wgpu::ComputePipeline,

    // Tone curve pipelines
    pub tone_curve_scurve: wgpu::ComputePipeline,
    pub tone_curve_asymmetric: wgpu::ComputePipeline,

    // Color operations
    pub color_matrix: wgpu::ComputePipeline,
    pub apply_gains: wgpu::ComputePipeline,

    // Histogram operations
    pub histogram_accumulate: wgpu::ComputePipeline,
    pub histogram_clear: wgpu::ComputePipeline,

    // Colorspace conversions
    pub rgb_to_hsl: wgpu::ComputePipeline,
    pub hsl_to_rgb: wgpu::ComputePipeline,
    pub hsl_adjust: wgpu::ComputePipeline,

    // Utility operations
    pub clamp_range: wgpu::ComputePipeline,
    pub exposure_multiply: wgpu::ComputePipeline,
    pub shadow_lift: wgpu::ComputePipeline,
    pub highlight_compress: wgpu::ComputePipeline,
    pub cb_layers: wgpu::ComputePipeline,

    // Subsample for efficient analysis downloads
    pub subsample: wgpu::ComputePipeline,
    pub subsample_layout: wgpu::BindGroupLayout,

    // Cached bind group layouts for batched operations
    /// Layout for storage buffer (read-write) + uniform buffer pattern
    pub storage_uniform_layout: wgpu::BindGroupLayout,
    /// Layout for histogram operations (read-only pixels + 3 atomic histogram buffers)
    pub histogram_layout: wgpu::BindGroupLayout,
}

/// Create all compute pipelines from shader sources.
pub fn create_pipelines(device: &wgpu::Device) -> Result<GpuPipelines, GpuError> {
    // Load shader modules
    let inversion_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("inversion"),
        source: wgpu::ShaderSource::Wgsl(Shaders::INVERSION.into()),
    });

    let tone_curve_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("tone_curve"),
        source: wgpu::ShaderSource::Wgsl(Shaders::TONE_CURVE.into()),
    });

    let color_matrix_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("color_matrix"),
        source: wgpu::ShaderSource::Wgsl(Shaders::COLOR_MATRIX.into()),
    });

    let histogram_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("histogram"),
        source: wgpu::ShaderSource::Wgsl(Shaders::HISTOGRAM.into()),
    });

    let color_convert_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("color_convert"),
        source: wgpu::ShaderSource::Wgsl(Shaders::COLOR_CONVERT.into()),
    });

    let utility_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("utility"),
        source: wgpu::ShaderSource::Wgsl(Shaders::UTILITY.into()),
    });

    let cb_inversion_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cb_inversion"),
        source: wgpu::ShaderSource::Wgsl(Shaders::CB_INVERSION.into()),
    });

    let cb_layers_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cb_layers"),
        source: wgpu::ShaderSource::Wgsl(Shaders::CB_LAYERS.into()),
    });

    let subsample_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("subsample"),
        source: wgpu::ShaderSource::Wgsl(Shaders::SUBSAMPLE.into()),
    });

    // Create pipeline layouts
    let storage_uniform_layout = create_storage_uniform_layout(device);
    let storage_uniform_pipeline_layout =
        create_storage_uniform_pipeline_layout(device, &storage_uniform_layout);

    let histogram_layout = create_histogram_layout(device);
    let histogram_pipeline_layout = create_histogram_pipeline_layout(device, &histogram_layout);

    let subsample_layout = create_subsample_layout(device);
    let subsample_pipeline_layout = create_subsample_pipeline_layout(device, &subsample_layout);

    // Create all compute pipelines
    let inversion_linear = create_compute_pipeline(
        device,
        "inversion_linear",
        &storage_uniform_pipeline_layout,
        &inversion_module,
        "invert_linear",
    );

    let inversion_log = create_compute_pipeline(
        device,
        "inversion_log",
        &storage_uniform_pipeline_layout,
        &inversion_module,
        "invert_log",
    );

    let inversion_divide = create_compute_pipeline(
        device,
        "inversion_divide",
        &storage_uniform_pipeline_layout,
        &inversion_module,
        "invert_divide",
    );

    let inversion_mask_aware = create_compute_pipeline(
        device,
        "inversion_mask_aware",
        &storage_uniform_pipeline_layout,
        &inversion_module,
        "invert_mask_aware",
    );

    let inversion_bw = create_compute_pipeline(
        device,
        "inversion_bw",
        &storage_uniform_pipeline_layout,
        &inversion_module,
        "invert_bw",
    );

    let cb_inversion = create_compute_pipeline(
        device,
        "cb_inversion",
        &storage_uniform_pipeline_layout,
        &cb_inversion_module,
        "invert_cb",
    );

    let tone_curve_scurve = create_compute_pipeline(
        device,
        "tone_curve_scurve",
        &storage_uniform_pipeline_layout,
        &tone_curve_module,
        "apply_scurve",
    );

    let tone_curve_asymmetric = create_compute_pipeline(
        device,
        "tone_curve_asymmetric",
        &storage_uniform_pipeline_layout,
        &tone_curve_module,
        "apply_asymmetric",
    );

    let color_matrix = create_compute_pipeline(
        device,
        "color_matrix",
        &storage_uniform_pipeline_layout,
        &color_matrix_module,
        "apply_color_matrix",
    );

    let apply_gains = create_compute_pipeline(
        device,
        "apply_gains",
        &storage_uniform_pipeline_layout,
        &color_matrix_module,
        "apply_gains",
    );

    let histogram_accumulate = create_compute_pipeline(
        device,
        "histogram_accumulate",
        &histogram_pipeline_layout,
        &histogram_module,
        "accumulate_histogram",
    );

    let histogram_clear = create_compute_pipeline(
        device,
        "histogram_clear",
        &histogram_pipeline_layout,
        &histogram_module,
        "clear_histogram",
    );

    let rgb_to_hsl = create_compute_pipeline(
        device,
        "rgb_to_hsl",
        &storage_uniform_pipeline_layout,
        &color_convert_module,
        "rgb_to_hsl",
    );

    let hsl_to_rgb = create_compute_pipeline(
        device,
        "hsl_to_rgb",
        &storage_uniform_pipeline_layout,
        &color_convert_module,
        "hsl_to_rgb",
    );

    let hsl_adjust = create_compute_pipeline(
        device,
        "hsl_adjust",
        &storage_uniform_pipeline_layout,
        &color_convert_module,
        "apply_hsl_adjustments",
    );

    let clamp_range = create_compute_pipeline(
        device,
        "clamp_range",
        &storage_uniform_pipeline_layout,
        &utility_module,
        "clamp_range",
    );

    let exposure_multiply = create_compute_pipeline(
        device,
        "exposure_multiply",
        &storage_uniform_pipeline_layout,
        &utility_module,
        "exposure_multiply",
    );

    let shadow_lift = create_compute_pipeline(
        device,
        "shadow_lift",
        &storage_uniform_pipeline_layout,
        &utility_module,
        "shadow_lift",
    );

    let highlight_compress = create_compute_pipeline(
        device,
        "highlight_compress",
        &storage_uniform_pipeline_layout,
        &utility_module,
        "highlight_compress",
    );

    let cb_layers = create_compute_pipeline(
        device,
        "cb_layers",
        &storage_uniform_pipeline_layout,
        &cb_layers_module,
        "apply_cb_layers",
    );

    let subsample = create_compute_pipeline(
        device,
        "subsample",
        &subsample_pipeline_layout,
        &subsample_module,
        "subsample",
    );

    Ok(GpuPipelines {
        inversion_linear,
        inversion_log,
        inversion_divide,
        inversion_mask_aware,
        inversion_bw,
        cb_inversion,
        tone_curve_scurve,
        tone_curve_asymmetric,
        color_matrix,
        apply_gains,
        histogram_accumulate,
        histogram_clear,
        rgb_to_hsl,
        hsl_to_rgb,
        hsl_adjust,
        clamp_range,
        exposure_multiply,
        shadow_lift,
        highlight_compress,
        cb_layers,
        subsample,
        subsample_layout,
        storage_uniform_layout,
        histogram_layout,
    })
}

/// Create a compute pipeline with the given parameters.
fn create_compute_pipeline(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
    entry_point: &str,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        module,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// Create the storage + uniform bind group layout.
fn create_storage_uniform_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("storage_uniform_layout"),
        entries: &[
            // Storage buffer (read-write pixels)
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
            // Uniform buffer (parameters)
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
    })
}

/// Create the storage + uniform pipeline layout.
fn create_storage_uniform_pipeline_layout(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::PipelineLayout {
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("storage_uniform_pipeline_layout"),
        bind_group_layouts: &[layout],
        push_constant_ranges: &[],
    })
}

/// Create the histogram bind group layout.
fn create_histogram_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("histogram_layout"),
        entries: &[
            // Storage buffer (read-only pixels)
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
            // Histogram R (atomic)
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
            // Histogram G (atomic)
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
            // Histogram B (atomic)
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
    })
}

/// Create the histogram pipeline layout.
fn create_histogram_pipeline_layout(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::PipelineLayout {
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("histogram_pipeline_layout"),
        bind_group_layouts: &[layout],
        push_constant_ranges: &[],
    })
}

/// Create the subsample bind group layout.
fn create_subsample_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("subsample_layout"),
        entries: &[
            // Input pixels (read-only)
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
            // Output pixels (read-write)
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
            // Parameters (uniform)
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

/// Create the subsample pipeline layout.
fn create_subsample_pipeline_layout(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::PipelineLayout {
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("subsample_pipeline_layout"),
        bind_group_layouts: &[layout],
        push_constant_ranges: &[],
    })
}

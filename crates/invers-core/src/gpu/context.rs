//! GPU context management for wgpu device, queue, and compute pipelines.

use std::sync::Arc;
use wgpu;

use super::shaders::Shaders;

/// Errors that can occur during GPU operations.
#[derive(Debug, Clone)]
pub enum GpuError {
    /// No suitable GPU adapter found
    NoAdapter,
    /// Failed to request GPU device
    DeviceRequest(String),
    /// Shader compilation failed
    ShaderCompilation(String),
    /// Buffer operation failed
    BufferError(String),
    /// Pipeline creation failed
    PipelineError(String),
    /// GPU execution failed
    ExecutionError(String),
    /// Other GPU-related error
    Other(String),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::NoAdapter => write!(f, "No suitable GPU adapter found"),
            GpuError::DeviceRequest(e) => write!(f, "Failed to request GPU device: {}", e),
            GpuError::ShaderCompilation(e) => write!(f, "Shader compilation failed: {}", e),
            GpuError::BufferError(e) => write!(f, "Buffer operation failed: {}", e),
            GpuError::PipelineError(e) => write!(f, "Pipeline creation failed: {}", e),
            GpuError::ExecutionError(e) => write!(f, "GPU execution failed: {}", e),
            GpuError::Other(e) => write!(f, "GPU error: {}", e),
        }
    }
}

impl std::error::Error for GpuError {}

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
}

/// GPU context holding the wgpu device, queue, and pre-compiled pipelines.
pub struct GpuContext {
    pub(crate) device: Arc<wgpu::Device>,
    pub(crate) queue: Arc<wgpu::Queue>,
    pub(crate) pipelines: GpuPipelines,
    adapter_info: wgpu::AdapterInfo,
}

impl GpuContext {
    /// Check if GPU acceleration is available without fully initializing.
    pub fn is_available() -> bool {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        pollster::block_on(async {
            instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .is_some()
        })
    }

    /// Get information about the available GPU device.
    pub fn device_info() -> Option<String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        pollster::block_on(async {
            instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .map(|adapter| {
                    let info = adapter.get_info();
                    format!("{} ({:?}, {:?})", info.name, info.device_type, info.backend)
                })
        })
    }

    /// Create a new GPU context, initializing the device and compiling all shaders.
    pub fn new() -> Result<Self, GpuError> {
        pollster::block_on(Self::new_async())
    }

    /// Async version of context creation.
    pub async fn new_async() -> Result<Self, GpuError> {
        // Create wgpu instance with all backends
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Request high-performance adapter
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(GpuError::NoAdapter)?;

        let adapter_info = adapter.get_info();

        // Get adapter limits to request maximum available
        let adapter_limits = adapter.limits();

        // Request device with required features and higher buffer limits for large images
        // Large scans (e.g., 4000x6000 @ 48-bit) can exceed 200MB
        let limits = wgpu::Limits {
            // Request max storage buffer size from adapter (for image data)
            max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
            // Also increase uniform buffer size if needed
            max_uniform_buffer_binding_size: adapter_limits.max_uniform_buffer_binding_size,
            // Increase buffer size limit
            max_buffer_size: adapter_limits.max_buffer_size,
            ..Default::default()
        };

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("invers-gpu"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| GpuError::DeviceRequest(e.to_string()))?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        // Compile all shaders and create pipelines
        let pipelines = Self::create_pipelines(&device)?;

        Ok(Self {
            device,
            queue,
            pipelines,
            adapter_info,
        })
    }

    /// Get the adapter info for this context.
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    /// Create all compute pipelines from shader sources.
    fn create_pipelines(device: &wgpu::Device) -> Result<GpuPipelines, GpuError> {
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

        // Create pipeline layout for storage buffer + uniform operations
        let storage_uniform_layout =
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
            });

        let storage_uniform_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("storage_uniform_pipeline_layout"),
                bind_group_layouts: &[&storage_uniform_layout],
                push_constant_ranges: &[],
            });

        // Create histogram pipeline layout (read-only pixels + atomic histogram buffers)
        let histogram_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        });

        let histogram_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("histogram_pipeline_layout"),
                bind_group_layouts: &[&histogram_layout],
                push_constant_ranges: &[],
            });

        // Create storage-only pipeline layout for histogram clear
        let storage_only_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("storage_only_layout"),
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
                ],
            });

        // Storage-only pipeline layout is currently unused but kept for potential future use
        let _storage_only_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("storage_only_pipeline_layout"),
                bind_group_layouts: &[&storage_only_layout],
                push_constant_ranges: &[],
            });

        // Create subsample pipeline layout (read-only input, read-write output, uniform params)
        let subsample_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        });

        let subsample_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("subsample_pipeline_layout"),
                bind_group_layouts: &[&subsample_layout],
                push_constant_ranges: &[],
            });

        // Create all compute pipelines
        let inversion_linear = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("inversion_linear"),
            layout: Some(&storage_uniform_pipeline_layout),
            module: &inversion_module,
            entry_point: Some("invert_linear"),
            compilation_options: Default::default(),
            cache: None,
        });

        let inversion_log = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("inversion_log"),
            layout: Some(&storage_uniform_pipeline_layout),
            module: &inversion_module,
            entry_point: Some("invert_log"),
            compilation_options: Default::default(),
            cache: None,
        });

        let inversion_divide = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("inversion_divide"),
            layout: Some(&storage_uniform_pipeline_layout),
            module: &inversion_module,
            entry_point: Some("invert_divide"),
            compilation_options: Default::default(),
            cache: None,
        });

        let inversion_mask_aware =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("inversion_mask_aware"),
                layout: Some(&storage_uniform_pipeline_layout),
                module: &inversion_module,
                entry_point: Some("invert_mask_aware"),
                compilation_options: Default::default(),
                cache: None,
            });

        let inversion_bw = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("inversion_bw"),
            layout: Some(&storage_uniform_pipeline_layout),
            module: &inversion_module,
            entry_point: Some("invert_bw"),
            compilation_options: Default::default(),
            cache: None,
        });

        let cb_inversion = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cb_inversion"),
            layout: Some(&storage_uniform_pipeline_layout),
            module: &cb_inversion_module,
            entry_point: Some("invert_cb"),
            compilation_options: Default::default(),
            cache: None,
        });

        let tone_curve_scurve = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tone_curve_scurve"),
            layout: Some(&storage_uniform_pipeline_layout),
            module: &tone_curve_module,
            entry_point: Some("apply_scurve"),
            compilation_options: Default::default(),
            cache: None,
        });

        let tone_curve_asymmetric =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("tone_curve_asymmetric"),
                layout: Some(&storage_uniform_pipeline_layout),
                module: &tone_curve_module,
                entry_point: Some("apply_asymmetric"),
                compilation_options: Default::default(),
                cache: None,
            });

        let color_matrix = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("color_matrix"),
            layout: Some(&storage_uniform_pipeline_layout),
            module: &color_matrix_module,
            entry_point: Some("apply_color_matrix"),
            compilation_options: Default::default(),
            cache: None,
        });

        let apply_gains = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apply_gains"),
            layout: Some(&storage_uniform_pipeline_layout),
            module: &color_matrix_module,
            entry_point: Some("apply_gains"),
            compilation_options: Default::default(),
            cache: None,
        });

        let histogram_accumulate =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("histogram_accumulate"),
                layout: Some(&histogram_pipeline_layout),
                module: &histogram_module,
                entry_point: Some("accumulate_histogram"),
                compilation_options: Default::default(),
                cache: None,
            });

        let histogram_clear = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("histogram_clear"),
            layout: Some(&histogram_pipeline_layout),
            module: &histogram_module,
            entry_point: Some("clear_histogram"),
            compilation_options: Default::default(),
            cache: None,
        });

        let rgb_to_hsl = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rgb_to_hsl"),
            layout: Some(&storage_uniform_pipeline_layout),
            module: &color_convert_module,
            entry_point: Some("rgb_to_hsl"),
            compilation_options: Default::default(),
            cache: None,
        });

        let hsl_to_rgb = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hsl_to_rgb"),
            layout: Some(&storage_uniform_pipeline_layout),
            module: &color_convert_module,
            entry_point: Some("hsl_to_rgb"),
            compilation_options: Default::default(),
            cache: None,
        });

        let hsl_adjust = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hsl_adjust"),
            layout: Some(&storage_uniform_pipeline_layout),
            module: &color_convert_module,
            entry_point: Some("apply_hsl_adjustments"),
            compilation_options: Default::default(),
            cache: None,
        });

        let clamp_range = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("clamp_range"),
            layout: Some(&storage_uniform_pipeline_layout),
            module: &utility_module,
            entry_point: Some("clamp_range"),
            compilation_options: Default::default(),
            cache: None,
        });

        let exposure_multiply = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("exposure_multiply"),
            layout: Some(&storage_uniform_pipeline_layout),
            module: &utility_module,
            entry_point: Some("exposure_multiply"),
            compilation_options: Default::default(),
            cache: None,
        });

        let shadow_lift = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("shadow_lift"),
            layout: Some(&storage_uniform_pipeline_layout),
            module: &utility_module,
            entry_point: Some("shadow_lift"),
            compilation_options: Default::default(),
            cache: None,
        });

        let highlight_compress = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("highlight_compress"),
            layout: Some(&storage_uniform_pipeline_layout),
            module: &utility_module,
            entry_point: Some("highlight_compress"),
            compilation_options: Default::default(),
            cache: None,
        });

        let cb_layers = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cb_layers"),
            layout: Some(&storage_uniform_pipeline_layout),
            module: &cb_layers_module,
            entry_point: Some("apply_cb_layers"),
            compilation_options: Default::default(),
            cache: None,
        });

        let subsample = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("subsample"),
            layout: Some(&subsample_pipeline_layout),
            module: &subsample_module,
            entry_point: Some("subsample"),
            compilation_options: Default::default(),
            cache: None,
        });

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
        })
    }

    /// Submit a command encoder and wait for completion.
    pub fn submit_and_wait(&self, encoder: wgpu::CommandEncoder) {
        self.queue.submit(std::iter::once(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
    }
}

//! GPU context management for wgpu device, queue, and compute pipelines.

mod init;
mod pipelines;

use std::sync::Arc;

pub use pipelines::GpuPipelines;

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
        init::is_available()
    }

    /// Get information about the available GPU device.
    pub fn device_info() -> Option<String> {
        init::device_info()
    }

    /// Create a new GPU context, initializing the device and compiling all shaders.
    pub fn new() -> Result<Self, GpuError> {
        pollster::block_on(Self::new_async())
    }

    /// Async version of context creation.
    pub async fn new_async() -> Result<Self, GpuError> {
        let (device, queue, adapter_info) = init::initialize_device().await?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        // Compile all shaders and create pipelines
        let pipelines = pipelines::create_pipelines(&device)?;

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

    /// Submit a command encoder and wait for completion.
    pub fn submit_and_wait(&self, encoder: wgpu::CommandEncoder) {
        self.queue.submit(std::iter::once(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
    }
}

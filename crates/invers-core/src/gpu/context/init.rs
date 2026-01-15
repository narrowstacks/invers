//! Device and adapter initialization for GPU context.

use super::GpuError;

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

/// Initialize the wgpu device and queue.
///
/// Returns the device, queue, and adapter info.
pub async fn initialize_device() -> Result<(wgpu::Device, wgpu::Queue, wgpu::AdapterInfo), GpuError>
{
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

    Ok((device, queue, adapter_info))
}

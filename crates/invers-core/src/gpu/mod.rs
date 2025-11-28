//! GPU acceleration module for image processing operations.
//!
//! This module provides GPU-accelerated versions of the core image processing
//! operations using wgpu (WebGPU). It supports Metal on macOS, Vulkan on Linux/Windows,
//! and DX12 on Windows.
//!
//! # Usage
//!
//! The GPU backend is enabled via the `gpu` feature flag:
//!
//! ```toml
//! [dependencies]
//! invers-core = { version = "0.1", features = ["gpu"] }
//! ```
//!
//! At runtime, GPU processing can be enabled/disabled via `ConvertOptions::use_gpu`.
//!
//! # Performance
//!
//! The GPU context (including compiled shaders) is cached globally and reused
//! across all image conversions. The first GPU operation will be slower due to
//! shader compilation, but subsequent operations reuse the cached pipelines.

mod context;
mod buffers;
mod pipeline;
mod shaders;

use std::sync::OnceLock;

pub use context::{GpuContext, GpuError};
pub use buffers::GpuImage;
pub use pipeline::process_image_gpu;

/// Global cached GPU context for reuse across operations.
/// Initialized lazily on first use, then reused for all subsequent GPU operations.
static GPU_CONTEXT: OnceLock<Result<GpuContext, GpuError>> = OnceLock::new();

/// Get a reference to the cached GPU context, initializing it if needed.
///
/// The first call will initialize the GPU device and compile all shaders,
/// which may take 100-500ms. Subsequent calls return immediately.
pub fn get_cached_context() -> Result<&'static GpuContext, GpuError> {
    GPU_CONTEXT
        .get_or_init(|| GpuContext::new())
        .as_ref()
        .map_err(|e| e.clone())
}

/// Pre-warm the GPU context by initializing it in the background.
///
/// Call this early in your application startup to avoid latency on first use.
/// This is non-blocking if called from an async context, but blocks if
/// the context is already being initialized by another thread.
pub fn prewarm_gpu() {
    let _ = get_cached_context();
}

/// Check if GPU acceleration is available on this system.
pub fn is_gpu_available() -> bool {
    GpuContext::is_available()
}

/// Get information about the available GPU device.
pub fn gpu_info() -> Option<String> {
    GpuContext::device_info()
}

#[cfg(test)]
mod tests;

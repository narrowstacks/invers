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

mod buffers;
mod context;
mod pipeline;
mod shaders;

pub use buffers::GpuImage;
pub use context::{GpuContext, GpuError};
pub use pipeline::process_image_gpu;

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

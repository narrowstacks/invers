//! GPU buffer management for image data and parameters.

use std::sync::Arc;
use wgpu::{self, util::DeviceExt};
use bytemuck::{Pod, Zeroable};

use super::context::GpuError;

/// Number of histogram buckets (16-bit precision).
pub const NUM_HISTOGRAM_BUCKETS: usize = 65536;

/// GPU image buffer with metadata.
pub struct GpuImage {
    pub(crate) buffer: wgpu::Buffer,
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

impl GpuImage {
    /// Create a new GPU image by uploading CPU data.
    pub fn upload(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        data: &[f32],
        width: u32,
        height: u32,
        channels: u32,
    ) -> Result<Self, GpuError> {
        let expected_size = (width * height * channels) as usize;
        if data.len() != expected_size {
            return Err(GpuError::BufferError(format!(
                "Data size mismatch: expected {}, got {}",
                expected_size,
                data.len()
            )));
        }

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu_image"),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        Ok(Self {
            buffer,
            width,
            height,
            channels,
            device,
            queue,
        })
    }

    /// Download the GPU image data back to CPU.
    pub fn download(&self) -> Result<Vec<f32>, GpuError> {
        let size = (self.width * self.height * self.channels) as u64 * std::mem::size_of::<f32>() as u64;

        // Create staging buffer for readback
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy from GPU buffer to staging
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("download_encoder"),
            });

        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging_buffer, 0, size);

        self.queue.submit(std::iter::once(encoder.finish()));

        // Map the staging buffer and read data
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        self.device.poll(wgpu::Maintain::Wait);

        rx.recv()
            .map_err(|e| GpuError::BufferError(e.to_string()))?
            .map_err(|e| GpuError::BufferError(e.to_string()))?;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        staging_buffer.unmap();

        Ok(result)
    }

    /// Get the total number of pixels.
    pub fn pixel_count(&self) -> u32 {
        self.width * self.height
    }

    /// Get the total number of f32 elements.
    pub fn element_count(&self) -> u32 {
        self.width * self.height * self.channels
    }
}

/// GPU histogram buffers for RGB channels.
pub struct GpuHistogram {
    pub(crate) buffer_r: wgpu::Buffer,
    pub(crate) buffer_g: wgpu::Buffer,
    pub(crate) buffer_b: wgpu::Buffer,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

impl GpuHistogram {
    /// Create new histogram buffers (initialized to zero).
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let buffer_size = (NUM_HISTOGRAM_BUCKETS * std::mem::size_of::<u32>()) as u64;

        let buffer_r = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("histogram_r"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let buffer_g = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("histogram_g"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let buffer_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("histogram_b"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            buffer_r,
            buffer_g,
            buffer_b,
            device,
            queue,
        }
    }

    /// Download histogram data to CPU for percentile computation.
    pub fn download(&self) -> Result<[Vec<u32>; 3], GpuError> {
        let size = (NUM_HISTOGRAM_BUCKETS * std::mem::size_of::<u32>()) as u64;

        // Create staging buffers
        let staging_r = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_hist_r"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_g = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_hist_g"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_b = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_hist_b"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy all histograms
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("histogram_download_encoder"),
            });

        encoder.copy_buffer_to_buffer(&self.buffer_r, 0, &staging_r, 0, size);
        encoder.copy_buffer_to_buffer(&self.buffer_g, 0, &staging_g, 0, size);
        encoder.copy_buffer_to_buffer(&self.buffer_b, 0, &staging_b, 0, size);

        self.queue.submit(std::iter::once(encoder.finish()));

        // Map and read all three
        let hist_r = Self::read_staging_buffer(&self.device, &staging_r)?;
        let hist_g = Self::read_staging_buffer(&self.device, &staging_g)?;
        let hist_b = Self::read_staging_buffer(&self.device, &staging_b)?;

        Ok([hist_r, hist_g, hist_b])
    }

    fn read_staging_buffer(
        device: &wgpu::Device,
        buffer: &wgpu::Buffer,
    ) -> Result<Vec<u32>, GpuError> {
        let buffer_slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        device.poll(wgpu::Maintain::Wait);

        rx.recv()
            .map_err(|e| GpuError::BufferError(e.to_string()))?
            .map_err(|e| GpuError::BufferError(e.to_string()))?;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        buffer.unmap();

        Ok(result)
    }
}

// Parameter structures for uniform buffers
// These must match the WGSL struct layouts exactly

/// Inversion parameters for uniform buffer.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct InversionParams {
    pub base_r: f32,
    pub base_g: f32,
    pub base_b: f32,
    pub green_floor: f32,
    pub blue_floor: f32,
    /// Headroom for B&W mode: fraction of base to preserve as shadow detail (e.g., 0.05 = 5%)
    pub bw_headroom: f32,
    pub pixel_count: u32,
    pub _padding: u32,
}

/// Tone curve parameters for uniform buffer.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ToneCurveParams {
    pub strength: f32,
    pub toe_strength: f32,
    pub toe_length: f32,
    pub shoulder_strength: f32,
    pub shoulder_start: f32,
    pub pixel_count: u32,
    pub _padding: [u32; 2],
}

/// Color matrix parameters for uniform buffer.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ColorMatrixParams {
    // Row-major 3x3 matrix with padding for alignment
    pub m00: f32,
    pub m01: f32,
    pub m02: f32,
    pub _pad0: f32,
    pub m10: f32,
    pub m11: f32,
    pub m12: f32,
    pub _pad1: f32,
    pub m20: f32,
    pub m21: f32,
    pub m22: f32,
    pub _pad2: f32,
    pub pixel_count: u32,
    pub _padding: [u32; 3],
}

/// Gain parameters for auto-levels/color/exposure.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GainParams {
    pub gain_r: f32,
    pub gain_g: f32,
    pub gain_b: f32,
    pub offset_r: f32,
    pub offset_g: f32,
    pub offset_b: f32,
    pub pixel_count: u32,
    pub _padding: u32,
}

/// HSL adjustment parameters for uniform buffer.
/// Uses vec4 pairs for 16-byte alignment required by WGSL uniform buffers.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct HslAdjustParams {
    // 8 color ranges split into vec4 pairs: R, O, Y, G (0) and A, B, P, M (1)
    pub hue_adj_0: [f32; 4],  // R, O, Y, G
    pub hue_adj_1: [f32; 4],  // A, B, P, M
    pub sat_adj_0: [f32; 4],  // R, O, Y, G
    pub sat_adj_1: [f32; 4],  // A, B, P, M
    pub lum_adj_0: [f32; 4],  // R, O, Y, G
    pub lum_adj_1: [f32; 4],  // A, B, P, M
    pub pixel_count: u32,
    pub _padding: [u32; 3],
}

/// Utility parameters for clamp, exposure, shadow lift, highlight compress.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct UtilityParams {
    pub param1: f32,
    pub param2: f32,
    pub param3: f32,
    pub pixel_count: u32,
}

/// Fused inversion parameters (invert + shadow lift + highlight compress).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct FusedInvertParams {
    // Inversion parameters
    pub base_r: f32,
    pub base_g: f32,
    pub base_b: f32,
    pub green_floor: f32,
    pub blue_floor: f32,
    pub bw_headroom: f32,

    // Shadow lift
    pub shadow_lift: f32,

    // Highlight compression
    pub highlight_threshold: f32,
    pub highlight_compression: f32,

    // Control flags (bits 0-2: mode, bit 3: shadow, bit 4: highlight)
    pub flags: u32,

    pub pixel_count: u32,
    pub _padding: u32,
}

impl FusedInvertParams {
    pub const MODE_LINEAR: u32 = 0;
    pub const MODE_LOG: u32 = 1;
    pub const MODE_DIVIDE: u32 = 2;
    pub const MODE_MASK_AWARE: u32 = 3;
    pub const MODE_BW: u32 = 4;

    pub const FLAG_SHADOW_LIFT: u32 = 8;      // bit 3
    pub const FLAG_HIGHLIGHT_COMPRESS: u32 = 16; // bit 4
}

/// Fused post-processing parameters (gains + matrix + tone curve + clamp).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct FusedPostprocessParams {
    // Gains (auto-levels)
    pub gain_r: f32,
    pub gain_g: f32,
    pub gain_b: f32,
    pub offset_r: f32,
    pub offset_g: f32,
    pub offset_b: f32,

    // Color matrix (row-major 3x3)
    pub m00: f32,
    pub m01: f32,
    pub m02: f32,
    pub m10: f32,
    pub m11: f32,
    pub m12: f32,
    pub m20: f32,
    pub m21: f32,
    pub m22: f32,

    // Tone curve
    pub tone_strength: f32,
    pub toe_strength: f32,
    pub toe_length: f32,
    pub shoulder_strength: f32,
    pub shoulder_start: f32,

    // Exposure
    pub exposure_multiplier: f32,

    // Control flags
    pub flags: u32,

    pub pixel_count: u32,
}

impl FusedPostprocessParams {
    pub const FLAG_GAINS: u32 = 1;           // bit 0
    pub const FLAG_COLOR_MATRIX: u32 = 2;    // bit 1
    pub const FLAG_TONE_SCURVE: u32 = 4;     // bit 2
    pub const FLAG_TONE_ASYMMETRIC: u32 = 8; // bit 3
    pub const FLAG_CLAMP: u32 = 16;          // bit 4
    pub const FLAG_EXPOSURE: u32 = 32;       // bit 5
}

/// Create a uniform buffer from parameter data.
pub fn create_uniform_buffer<T: Pod>(device: &wgpu::Device, data: &T, label: &str) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::bytes_of(data),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

/// Maximum uniform buffer size (must accommodate largest params struct)
pub const MAX_UNIFORM_SIZE: usize = 256;

/// Pool of reusable uniform buffers to avoid allocation overhead.
/// Each buffer is sized to accommodate any parameter struct.
pub struct UniformBufferPool {
    buffers: Vec<wgpu::Buffer>,
    next_index: usize,
}

impl UniformBufferPool {
    /// Create a new pool with the specified number of buffers.
    pub fn new(device: &wgpu::Device, count: usize) -> Self {
        let buffers = (0..count)
            .map(|i| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("uniform_pool_{}", i)),
                    size: MAX_UNIFORM_SIZE as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();

        Self {
            buffers,
            next_index: 0,
        }
    }

    /// Get the next available buffer from the pool (round-robin).
    /// Writes the data to the buffer using the queue.
    pub fn get_buffer<T: Pod>(
        &mut self,
        queue: &wgpu::Queue,
        data: &T,
    ) -> &wgpu::Buffer {
        let buffer = &self.buffers[self.next_index];
        queue.write_buffer(buffer, 0, bytemuck::bytes_of(data));
        self.next_index = (self.next_index + 1) % self.buffers.len();
        buffer
    }

    /// Reset the pool index for a new frame/operation.
    pub fn reset(&mut self) {
        self.next_index = 0;
    }
}

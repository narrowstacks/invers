//! GPU pipeline orchestration for image processing.

mod analysis;
mod cb;
mod dispatch;
mod histogram;
mod legacy;
mod ops;
mod ops_batched;

// Re-export public API
pub use cb::process_image_cb_gpu;
pub use legacy::process_image_gpu;

use wgpu;

use super::context::GpuContext;

/// Workgroup size for compute shaders
pub(crate) const WORKGROUP_SIZE: u32 = 256;

/// Maximum workgroups per dimension (GPU limit)
pub(crate) const MAX_WORKGROUPS_PER_DIM: u32 = 65535;

/// Maximum pixels per single dispatch (65535 workgroups * 256 threads)
#[allow(dead_code)]
pub(crate) const MAX_PIXELS_PER_DISPATCH: u32 = MAX_WORKGROUPS_PER_DIM * WORKGROUP_SIZE;

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
    pub(crate) ctx: &'a GpuContext,
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
    pub(crate) fn encoder_mut(&mut self) -> &mut wgpu::CommandEncoder {
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

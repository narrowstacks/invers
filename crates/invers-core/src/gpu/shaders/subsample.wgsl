// Subsample shader for extracting every Nth pixel from an image.
// Used to efficiently download a representative sample for analysis
// without transferring the entire image.

struct SubsampleParams {
    input_width: u32,
    input_height: u32,
    output_width: u32,
    stride: u32,
    output_pixel_count: u32,
    _padding1: u32,
    _padding2: u32,
    _padding3: u32,
}

@group(0) @binding(0) var<storage, read> input_pixels: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_pixels: array<f32>;
@group(0) @binding(2) var<uniform> params: SubsampleParams;

const WORKGROUP_SIZE: u32 = 256u;

fn get_pixel_index(id: vec3<u32>, num_workgroups: vec3<u32>) -> u32 {
    return id.y * num_workgroups.x * WORKGROUP_SIZE + id.x;
}

@compute @workgroup_size(256)
fn subsample(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let out_pixel_idx = get_pixel_index(id, num_workgroups);
    if (out_pixel_idx >= params.output_pixel_count) {
        return;
    }

    // Convert linear output index to 2D output coordinates
    let out_x = out_pixel_idx % params.output_width;
    let out_y = out_pixel_idx / params.output_width;

    // Map to input coordinates
    let in_x = out_x * params.stride;
    let in_y = out_y * params.stride;

    // Bounds check (should be guaranteed by output_pixel_count, but be safe)
    if (in_x >= params.input_width || in_y >= params.input_height) {
        return;
    }

    // Calculate buffer indices (3 channels per pixel)
    let in_idx = (in_y * params.input_width + in_x) * 3u;
    let out_idx = out_pixel_idx * 3u;

    // Copy RGB values
    output_pixels[out_idx] = input_pixels[in_idx];
    output_pixels[out_idx + 1u] = input_pixels[in_idx + 1u];
    output_pixels[out_idx + 2u] = input_pixels[in_idx + 2u];
}

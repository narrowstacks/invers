// Utility shaders for common operations.
// Includes clamp, exposure, shadow lift, and highlight compression.

struct UtilityParams {
    param1: f32,
    param2: f32,
    param3: f32,
    pixel_count: u32,
}

@group(0) @binding(0) var<storage, read_write> pixels: array<f32>;
@group(0) @binding(1) var<uniform> params: UtilityParams;

// Working range constants
const WORKING_RANGE_FLOOR: f32 = 1.0 / 65535.0;
const WORKING_RANGE_CEILING: f32 = 1.0 - WORKING_RANGE_FLOOR;

// Workgroup size for all shaders
const WORKGROUP_SIZE: u32 = 256u;

// Calculate linear pixel index from 2D dispatch grid
// Supports images larger than 65535 workgroups by using 2D dispatch
fn get_pixel_index(id: vec3<u32>, num_workgroups: vec3<u32>) -> u32 {
    return id.y * num_workgroups.x * WORKGROUP_SIZE + id.x;
}

// Clamp values to working range
// param1, param2, param3: unused
@compute @workgroup_size(256)
fn clamp_range(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let pixel_idx = get_pixel_index(id, num_workgroups);
    if (pixel_idx >= params.pixel_count) {
        return;
    }

    let idx = pixel_idx * 3u;

    pixels[idx] = clamp(pixels[idx], WORKING_RANGE_FLOOR, WORKING_RANGE_CEILING);
    pixels[idx + 1u] = clamp(pixels[idx + 1u], WORKING_RANGE_FLOOR, WORKING_RANGE_CEILING);
    pixels[idx + 2u] = clamp(pixels[idx + 2u], WORKING_RANGE_FLOOR, WORKING_RANGE_CEILING);
}

// Apply exposure multiplier
// param1: multiplier (exposure compensation)
// param2: max_value (for clamping, use 1.0 for normal, higher for no-clip mode)
// param3: unused
@compute @workgroup_size(256)
fn exposure_multiply(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let pixel_idx = get_pixel_index(id, num_workgroups);
    if (pixel_idx >= params.pixel_count) {
        return;
    }

    let idx = pixel_idx * 3u;
    let multiplier = params.param1;
    let max_val = params.param2;

    pixels[idx] = clamp(pixels[idx] * multiplier, 0.0, max_val);
    pixels[idx + 1u] = clamp(pixels[idx + 1u] * multiplier, 0.0, max_val);
    pixels[idx + 2u] = clamp(pixels[idx + 2u] * multiplier, 0.0, max_val);
}

// Apply uniform shadow lift
// param1: lift amount (added to all values)
// param2: unused
// param3: unused
@compute @workgroup_size(256)
fn shadow_lift(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let pixel_idx = get_pixel_index(id, num_workgroups);
    if (pixel_idx >= params.pixel_count) {
        return;
    }

    let idx = pixel_idx * 3u;
    let lift = params.param1;

    pixels[idx] = clamp(pixels[idx] + lift, 0.0, 1.0);
    pixels[idx + 1u] = clamp(pixels[idx + 1u] + lift, 0.0, 1.0);
    pixels[idx + 2u] = clamp(pixels[idx + 2u] + lift, 0.0, 1.0);
}

// Apply highlight compression (soft-clip above threshold)
// param1: threshold (e.g., 0.9)
// param2: compression factor (0.0 = full compress, 1.0 = no compress)
// param3: unused
@compute @workgroup_size(256)
fn highlight_compress(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let pixel_idx = get_pixel_index(id, num_workgroups);
    if (pixel_idx >= params.pixel_count) {
        return;
    }

    let idx = pixel_idx * 3u;
    let threshold = params.param1;
    let compression = params.param2;

    for (var c: u32 = 0u; c < 3u; c = c + 1u) {
        let value = pixels[idx + c];
        if (value > threshold) {
            let excess = value - threshold;
            pixels[idx + c] = threshold + excess * compression;
        }
    }
}

// Apply base offset addition (for film preset base_offsets)
// param1: offset_r
// param2: offset_g
// param3: offset_b
@compute @workgroup_size(256)
fn apply_base_offsets(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let pixel_idx = get_pixel_index(id, num_workgroups);
    if (pixel_idx >= params.pixel_count) {
        return;
    }

    let idx = pixel_idx * 3u;

    pixels[idx] = clamp(pixels[idx] + params.param1, 0.0, 1.0);
    pixels[idx + 1u] = clamp(pixels[idx + 1u] + params.param2, 0.0, 1.0);
    pixels[idx + 2u] = clamp(pixels[idx + 2u] + params.param3, 0.0, 1.0);
}

// Color matrix multiplication and gain application shaders.

// Working range constants
const WORKING_RANGE_FLOOR: f32 = 1.0 / 65535.0;
const WORKING_RANGE_CEILING: f32 = 1.0 - WORKING_RANGE_FLOOR;

// Soft-clip knee size (5% - matches CPU implementation)
const SOFT_CLIP_KNEE: f32 = 0.05;

struct ColorMatrixParams {
    // Row-major 3x3 matrix with padding for 16-byte alignment
    m00: f32, m01: f32, m02: f32, _pad0: f32,
    m10: f32, m11: f32, m12: f32, _pad1: f32,
    m20: f32, m21: f32, m22: f32, _pad2: f32,
    pixel_count: u32,
    _padding: vec3<u32>,
}

struct GainParams {
    gain_r: f32,
    gain_g: f32,
    gain_b: f32,
    offset_r: f32,
    offset_g: f32,
    offset_b: f32,
    pixel_count: u32,
    _padding: u32,
}

@group(0) @binding(0) var<storage, read_write> pixels: array<f32>;
@group(0) @binding(1) var<uniform> matrix_params: ColorMatrixParams;

// Workgroup size for all shaders
const WORKGROUP_SIZE: u32 = 256u;

// Calculate linear pixel index from 2D dispatch grid
// Supports images larger than 65535 workgroups by using 2D dispatch
fn get_pixel_index(id: vec3<u32>, num_workgroups: vec3<u32>) -> u32 {
    return id.y * num_workgroups.x * WORKGROUP_SIZE + id.x;
}

// Soft-clip a value to [0, 1] range with smooth knee
// Unlike hard clamping which loses all information above 1.0,
// soft-clipping applies a smooth compression curve that preserves
// relative differences while preventing overflow.
fn soft_clip_highlight(value: f32, knee: f32) -> f32 {
    if (value <= 0.0) {
        return WORKING_RANGE_FLOOR;
    } else if (value <= 1.0 - knee) {
        return value;
    } else if (value >= 1.0 + knee) {
        return WORKING_RANGE_CEILING;
    } else {
        let knee_start = 1.0 - knee;
        let normalized = (value - knee_start) / (2.0 * knee);
        let compressed = 1.0 - exp(-normalized * 2.0);
        return knee_start + knee * compressed;
    }
}

// Apply 3x3 color matrix to RGB pixels
// output = matrix × input
@compute @workgroup_size(256)
fn apply_color_matrix(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let pixel_idx = get_pixel_index(id, num_workgroups);
    if (pixel_idx >= matrix_params.pixel_count) {
        return;
    }

    let idx = pixel_idx * 3u;
    let r = pixels[idx];
    let g = pixels[idx + 1u];
    let b = pixels[idx + 2u];

    // Matrix multiplication
    let new_r = matrix_params.m00 * r + matrix_params.m01 * g + matrix_params.m02 * b;
    let new_g = matrix_params.m10 * r + matrix_params.m11 * g + matrix_params.m12 * b;
    let new_b = matrix_params.m20 * r + matrix_params.m21 * g + matrix_params.m22 * b;

    // Clamp to working range (no soft-clip to preserve highlight detail)
    pixels[idx] = clamp(new_r, WORKING_RANGE_FLOOR, WORKING_RANGE_CEILING);
    pixels[idx + 1u] = clamp(new_g, WORKING_RANGE_FLOOR, WORKING_RANGE_CEILING);
    pixels[idx + 2u] = clamp(new_b, WORKING_RANGE_FLOOR, WORKING_RANGE_CEILING);
}

// Separate binding for gain params (used in different entry point)
@group(0) @binding(1) var<uniform> gain_params: GainParams;

// Apply per-channel gains and offsets
// output = (input + offset) * gain
// Used for auto-levels, auto-color, auto-exposure application
@compute @workgroup_size(256)
fn apply_gains(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let pixel_idx = get_pixel_index(id, num_workgroups);
    if (pixel_idx >= gain_params.pixel_count) {
        return;
    }

    let idx = pixel_idx * 3u;

    // Apply offset then gain: (value - offset) * gain
    // For auto-levels: offset = min, gain = 1/(max-min)
    let r = (pixels[idx] - gain_params.offset_r) * gain_params.gain_r;
    let g = (pixels[idx + 1u] - gain_params.offset_g) * gain_params.gain_g;
    let b = (pixels[idx + 2u] - gain_params.offset_b) * gain_params.gain_b;

    pixels[idx] = clamp(r, 0.0, 1.0);
    pixels[idx + 1u] = clamp(g, 0.0, 1.0);
    pixels[idx + 2u] = clamp(b, 0.0, 1.0);
}

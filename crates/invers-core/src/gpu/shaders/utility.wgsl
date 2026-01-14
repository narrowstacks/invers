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

// Default soft-clip knee size (5%)
const SOFT_CLIP_KNEE: f32 = 0.05;

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
// Optimized with select() for reduced branching
fn soft_clip_highlight(value: f32, knee: f32) -> f32 {
    let knee_start = 1.0 - knee;

    // Compute all possible results
    let floor_result = WORKING_RANGE_FLOOR;
    let passthrough = value;
    let ceiling_result = WORKING_RANGE_CEILING;

    // Knee region: smooth exponential compression
    let normalized = (value - knee_start) / (2.0 * knee);
    let compressed = 1.0 - exp(-normalized * 2.0);
    let knee_result = knee_start + knee * compressed;

    // Select based on value region (branchless chained select)
    return select(
        select(
            select(ceiling_result, knee_result, value < 1.0 + knee),
            passthrough,
            value <= knee_start
        ),
        floor_result,
        value <= 0.0
    );
}

// Soft-clip with default 5% knee
fn soft_clip(value: f32) -> f32 {
    return soft_clip_highlight(value, SOFT_CLIP_KNEE);
}

// Clamp values to working range (hard clamp)
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

// Soft-clip values to working range (preserves highlight detail)
// param1: knee size (default 0.05 = 5%)
// param2, param3: unused
@compute @workgroup_size(256)
fn soft_clip_range(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let pixel_idx = get_pixel_index(id, num_workgroups);
    if (pixel_idx >= params.pixel_count) {
        return;
    }

    let idx = pixel_idx * 3u;
    let knee = select(SOFT_CLIP_KNEE, params.param1, params.param1 > 0.0);

    pixels[idx] = soft_clip_highlight(pixels[idx], knee);
    pixels[idx + 1u] = soft_clip_highlight(pixels[idx + 1u], knee);
    pixels[idx + 2u] = soft_clip_highlight(pixels[idx + 2u], knee);
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

// Helper function to compress a single channel value
fn compress_channel(value: f32, threshold: f32, compression: f32) -> f32 {
    let excess = value - threshold;
    let compressed = threshold + excess * compression;
    return select(value, compressed, value > threshold);
}

// Apply highlight compression (soft-clip above threshold)
// param1: threshold (e.g., 0.9)
// param2: compression factor (0.0 = full compress, 1.0 = no compress)
// param3: unused
// Optimized: unrolled loop and branchless compression
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

    // Unrolled loop with branchless compression using select()
    pixels[idx] = compress_channel(pixels[idx], threshold, compression);
    pixels[idx + 1u] = compress_channel(pixels[idx + 1u], threshold, compression);
    pixels[idx + 2u] = compress_channel(pixels[idx + 2u], threshold, compression);
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
